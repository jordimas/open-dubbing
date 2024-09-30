# Copyright 2024 Google LLC
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
import tempfile

from typing import Mapping
from unittest.mock import patch

import pytest

from pydub import AudioSegment

from open_dubbing.text_to_speech import TextToSpeech


class TextToSpeechUT(TextToSpeech):
    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        pitch: float,
        speed: float,
        volume_gain_db: float,
    ) -> str:
        pass

    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        pass

    def get_languages(self):
        pass


class TestCalculateTargetUtteranceSpeed:

    def test_calculate_target_utterance_speed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dubbed_audio_mock = AudioSegment.silent(duration=90000)
            dubbed_file_path = os.path.join(tempdir, "dubbed.mp3")
            dubbed_audio_mock.export(dubbed_file_path, format="mp3")
            result = TextToSpeechUT()._calculate_target_utterance_speed(
                reference_length=60.0, dubbed_file=dubbed_file_path
            )
            expected_result = 90000 / 60000
            assert result == expected_result


class TestCreateSpeakerToPathsMapping:

    @pytest.mark.parametrize(
        "input_data, expected_result",
        [
            ([], {}),
            (
                [
                    {"speaker_id": "speaker1", "vocals_path": "path/to/file1.wav"},
                    {"speaker_id": "speaker1", "vocals_path": "path/to/file2.wav"},
                ],
                {"speaker1": ["path/to/file1.wav", "path/to/file2.wav"]},
            ),
            (
                [
                    {"speaker_id": "speaker1", "vocals_path": "path/to/file1.wav"},
                    {"speaker_id": "speaker2", "vocals_path": "path/to/file3.wav"},
                    {"speaker_id": "speaker1", "vocals_path": "path/to/file2.wav"},
                ],
                {
                    "speaker1": ["path/to/file1.wav", "path/to/file2.wav"],
                    "speaker2": ["path/to/file3.wav"],
                },
            ),
        ],
    )
    def test_create_speaker_to_paths_mapping(self, input_data, expected_result):
        result = TextToSpeechUT().create_speaker_to_paths_mapping(input_data)
        assert result == expected_result


class TestTextToSpeech:

    @pytest.mark.parametrize(
        "calculated_speed, expect_adjust_called, expected_final_speed",
        [
            (0.5, False, 1.0),  # Speed below 1.0 is 1.0
            (1.2, True, 1.2),  # Speed at 1.2, should adjust speed
            (1.5, True, 1.3),  # Speed exceeds 1.2, clamped to max 1.2
        ],
    )
    def test_dub_utterances_with_speeds(
        self, calculated_speed, expect_adjust_called, expected_final_speed
    ):
        #        assert False == True
        tts = TextToSpeechUT()  # Your actual TextToSpeech implementation

        # Mock dependencies
        utterance_metadata = [
            {
                "for_dubbing": True,
                "assigned_voice": "en_voice",
                "start": 0,
                "end": 5,
                "translated_text": "Hello world",
                "pitch": 1.0,
                "speed": 1.0,  # Initially set speed to 1.0
                "volume_gain_db": 0.0,
                "path": "some/path/file.mp3",
            }
        ]
        output_directory = "/output"
        target_language = "en"

        # Mock methods
        with patch.object(
            tts, "_does_voice_supports_speeds", return_value=False
        ), patch.object(
            tts, "_convert_text_to_speech", return_value="dubbed_file_path"
        ), patch.object(
            tts, "_adjust_audio_speed"
        ) as mock_adjust_speed, patch.object(
            tts, "_calculate_target_utterance_speed", return_value=calculated_speed
        ):
            result = tts.dub_utterances(
                utterance_metadata=utterance_metadata,
                output_directory=output_directory,
                target_language=target_language,
            )

            if expect_adjust_called:
                mock_adjust_speed.assert_called_once()
            else:
                mock_adjust_speed.assert_not_called()

            assert result[0]["speed"] == expected_final_speed
