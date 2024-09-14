# Copyright 2024 Google LLC
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

"""Tests for utility functions in TextToSpeechMMS().py."""

import os
import tempfile
import pytest
from open_dubbing.text_to_speech_mms import TextToSpeechMMS
from pydub import AudioSegment


class TestCalculateTargetUtteranceSpeed:

    def test_calculate_target_utterance_speed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dubbed_audio_mock = AudioSegment.silent(duration=90000)
            dubbed_file_path = os.path.join(tempdir, "dubbed.mp3")
            dubbed_audio_mock.export(dubbed_file_path, format="mp3")
            result = TextToSpeechMMS()._calculate_target_utterance_speed(
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
        result = TextToSpeechMMS().create_speaker_to_paths_mapping(input_data)
        assert result == expected_result
