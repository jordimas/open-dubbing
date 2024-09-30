# Copyright 2024 Jordi Mas i Hernàndez <jmas@softcatala.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydub import AudioSegment
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(
            config, 3
        )  # Original model has 3 classes, we'll handle 2
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


class VoiceGenderClassifier:

    def __init__(self):

        # Load model from hub
        self.device = "cpu"
        model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name)

    # Function to load and process the MP3 file using pydub
    def load_audio_file(self, file_path, target_sampling_rate=16000):
        max_duration = 10  # Max duration in seconds

        # Load the audio file using pydub
        audio = AudioSegment.from_file(file_path, format="mp3")

        # If audio is longer than max_duration, trim it
        if len(audio) > max_duration * 1000:  # Convert seconds to milliseconds
            audio = audio[: max_duration * 1000]

        # Resample the audio to target sampling rate
        audio = audio.set_frame_rate(target_sampling_rate)

        # Convert to numpy array (pydub uses samples in the form of raw bytes)
        samples = np.array(audio.get_array_of_samples())

        # Ensure mono audio (convert stereo to mono if needed)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # Average the two channels

        # Normalize the audio samples to the range [-1, 1] (convert to float32)
        samples = samples.astype(np.float32) / np.iinfo(np.int16).max

        # Ensure audio is in the correct shape (1, length)
        samples = np.expand_dims(samples, axis=0)

        return samples, target_sampling_rate

    def _predict(
        self,
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
    ) -> np.ndarray:
        r"""Predict age and gender or extract embeddings from raw audio signal."""

        # Run through processor to normalize signal
        y = self.processor(x, sampling_rate=sampling_rate)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # Run through model
        with torch.no_grad():
            y = self.model(y)
            if embeddings:
                return y[0]  # Return hidden states (embeddings)
            else:
                logits_age, logits_gender = y[1], y[2]  # Age and gender logits
                return logits_age, logits_gender

    def _interpret_gender(self, logits_gender):
        """Convert gender logits into a male/female label."""
        # We're only interested in male and female, so we take the first two logits (ignore "child")
        male_female_logits = logits_gender[
            :, :2
        ]  # Take the first two logits (female, male)

        # Gender labels with only male and female
        gender_labels = ["Female", "Male"]

        # Find the index with the highest probability between female and male
        predicted_gender_idx = torch.argmax(male_female_logits, dim=1).item()
        probabilities = F.softmax(male_female_logits, dim=1)
        prob_female, prob_male = probabilities[0].tolist()  # Since it's batch size of 1

        return gender_labels[predicted_gender_idx]

    def get_gender_for_file(self, file_path):
        signal, sampling_rate = self.load_audio_file(file_path)

        # Predict age and gender
        age_logits, gender_logits = self._predict(signal, sampling_rate)
        predicted_gender = self._interpret_gender(gender_logits)
        logging.debug(
            f"The audio from {os.path.basename(file_path)} is predicted {predicted_gender}"
        )
        return predicted_gender
