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

import array
import logging

import numpy as np
import torch

from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from open_dubbing.speech_to_text import SpeechToText


class SpeechToTextWhisperTransfomers(SpeechToText):

    def __init__(self, device="cpu", cpu_threads=0):
        super().__init__(device, cpu_threads)
        self._processor = None

    def load_model(self):
        model_name = "openai/whisper-medium"
        self._processor = WhisperProcessor.from_pretrained(model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def _transcribe(
        self,
        *,
        vocals_filepath: str,
        source_language_iso_639_1: str,
    ) -> str:
        audio = AudioSegment.from_file(vocals_filepath)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set the frame rate to 16kHz

        MIN_SECS = 0.5
        # To prevent HuggingFace Whisper hallucinations with very short audios
        if len(audio) < 1000 * MIN_SECS:
            logging.warn(
                f"speech_to_text_whisper_transfomers._transcribe. Audio is less than {MIN_SECS} second, skipping transcription of '{vocals_filepath}'."
            )
            return ""

        # Convert the audio to a numpy array
        audio_input = (
            np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        )  # Normalize

        # Preprocess the audio input
        input_features = self._processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_features, language=source_language_iso_639_1
            )
        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        logging.debug(
            f"speech_to_text_whisper_transfomers._transcribe. transcription: {transcription}, file {vocals_filepath}"
        )
        return transcription

    def _get_audio_language(self, audio: array.array) -> str:
        audio_input = np.array(audio).astype(np.float32) / 32768.0

        # Preprocess the audio input
        input_features = self._processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            generated_ids = self._model.generate(input_features)

        # Decode the transcription including special tokens to capture the language token
        transcription_with_tokens = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        detected_language = None
        if "|" in transcription_with_tokens:
            for token in transcription_with_tokens.split("|"):
                if (len(token) == 2 or len(token) == 3) and token.isalpha():
                    detected_language = self._get_iso_639_3(token)
                    break
        logging.debug(
            f"speech_to_text_whisper_transfomers._get_audio_language. Detected language: {detected_language}"
        )

        return detected_language

    def get_languages(self):
        languages = [
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
            "yue",
        ]

        iso_639_3 = []
        for language in languages:
            pt3 = self._get_iso_639_3(language)
            iso_639_3.append(pt3)
        return iso_639_3
