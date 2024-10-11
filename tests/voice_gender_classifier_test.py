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

import os

import numpy as np
import pytest
import torch

from open_dubbing.voice_gender_classifier import (  # Assuming this is the file name
    VoiceGenderClassifier,
)


class TestVoiceGenderClassifier:

    @pytest.mark.parametrize(
        "logits_gender, expected_gender",
        [
            (torch.tensor([[2.0, 1.0]]), "Female"),  # Higher for female
            (torch.tensor([[1.0, 2.0]]), "Male"),  # Higher for male
        ],
    )
    def test_interpret_gender(self, logits_gender, expected_gender):
        classifier = VoiceGenderClassifier()
        predicted_gender = classifier._interpret_gender(logits_gender)

        assert predicted_gender == expected_gender

    def test_load_audio_file(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")

        classifier = VoiceGenderClassifier()
        samples, target_sampling_rate = classifier.load_audio_file(filename)
        sample_sum = np.sum(samples)
        assert 16000 == target_sampling_rate
        assert np.isclose(sample_sum, -19.797165, atol=2)

    def test_get_gender_for_file(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")

        classifier = VoiceGenderClassifier()
        gender = classifier.get_gender_for_file(filename)
        assert "Male" == gender
