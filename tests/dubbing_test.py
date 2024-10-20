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

"""Tests for utility functions in dubbing.py."""

import json
import os
import tempfile

import pytest

from open_dubbing import dubbing
from open_dubbing.dubbing import Dubber


class TestDubbing:

    @pytest.mark.parametrize(
        "original_file, expected_result",
        [
            (
                "Test File With Spaces and Uppercase.mp4",
                "testfilewithspacesanduppercase.mp4",
            ),
            ("Test-File-2024-with-Hyphens.mov", "testfile2024withhyphens.mov"),
            ("lowercasefilename.avi", "lowercasefilename.avi"),
        ],
    )
    def test_rename(self, original_file, expected_result):
        result = dubbing.rename_input_file(original_file)
        assert result == expected_result

    @pytest.mark.parametrize(
        "original_filename, expected_filename",
        [
            (
                "Test File With Spaces and Uppercase.mp4",
                "testfilewithspacesanduppercase.mp4",
            ),
            ("Test-File-2024-with-Hyphens.mov", "testfile2024withhyphens.mov"),
            ("lowercasefilename.avi", "lowercasefilename.avi"),
        ],
    )
    def test_overwrite_input_file(self, original_filename, expected_filename):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_full_path = os.path.join(temp_dir, original_filename)
            expected_full_path = os.path.join(temp_dir, expected_filename)

            with open(original_full_path, "w") as f:
                f.write("Test content")

            dubbing.overwrite_input_file(original_full_path, expected_full_path)
            assert os.path.exists(expected_full_path)

    def testrun_save_utterance_metadata(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = temp_dir

            dubbing = Dubber(
                input_file="",
                output_directory=directory,
                source_language="spa",
                target_language="cat",
                target_language_region="",
                hugging_face_token="",
                tts=None,
                translation=None,
                stt=None,
                device="cpu",
            )
            dubbing.utterance_metadata = {
                "utterances": {
                    "utterances": [
                        {"start": 1.26, "end": 3.94},
                        {"start": 5.24, "end": 6.629},
                    ]
                }
            }

            dubbing.run_save_utterance_metadata()
            metadata_file = os.path.join(directory, "utterance_metadata_cat.json")
            with open(metadata_file, encoding="utf-8") as json_data:
                data = json.load(json_data)

                print(data)
                assert data == {
                    "utterances": {
                        "utterances": {
                            "utterances": [
                                {"start": 1.26, "end": 3.94},
                                {"start": 5.24, "end": 6.629},
                            ]
                        }
                    },
                    "source_language": "spa",
                }
