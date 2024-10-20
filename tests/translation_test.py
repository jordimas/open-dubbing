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

"""Tests for utility functions in translation.py."""

import pytest

from open_dubbing.translation import Translation


class TranslationUT(Translation):
    def load_model(self):
        pass

    def get_language_pairs(self):
        return []

    def _translate_text(
        self, source_language: str, target_language: str, text: str
    ) -> str:
        return text


class TestTranslation:

    @pytest.mark.parametrize(
        "test_name, utterance_metadata, expected_script",
        [
            ("empty_input", [], "<BREAK><BREAK>"),
            (
                "single_utterance",
                [
                    {
                        "text": "Hello, world!",
                        "start": 0.0,
                        "stop": 1.0,
                        "speaker_id": "speaker1",
                        "ssml_gender": "male",
                    }
                ],
                "<BREAK>Hello, world!<BREAK>",
            ),
            (
                "multiple_utterances",
                [
                    {
                        "text": "This is",
                        "start": 0.0,
                        "stop": 1.0,
                        "speaker_id": "speaker1",
                        "ssml_gender": "male",
                    },
                    {
                        "text": "a test.",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker2",
                        "ssml_gender": "female",
                    },
                ],
                "<BREAK>This is<BREAK>a test.<BREAK>",
            ),
            (
                "empty_string",
                [
                    {
                        "text": "This is",
                        "start": 0.0,
                        "stop": 1.0,
                        "speaker_id": "speaker1",
                        "ssml_gender": "male",
                    },
                    {
                        "text": "",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker2",
                        "ssml_gender": "female",
                    },
                ],
                "<BREAK>This is<BREAK><BREAK>",
            ),
        ],
    )
    def test_generate_script(self, test_name, utterance_metadata, expected_script):
        result = TranslationUT()._generate_script(utterance_metadata=utterance_metadata)
        assert (
            result == expected_script
        ), f"Failed for {test_name}: expected {expected_script}, got {result}"

    @pytest.mark.parametrize(
        "utterance_metadata, translated_script, expected_translated_metadata",
        [
            (
                [
                    {
                        "text": "Hello",
                        "start": 0.0,
                        "stop": 1.0,
                        "speaker_id": "speaker1",
                        "ssml_gender": "male",
                    },
                    {
                        "text": "World",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker2",
                        "ssml_gender": "female",
                    },
                ],
                "Bonjour<BREAK>Le Monde",
                [
                    {
                        "text": "Hello",
                        "start": 0.0,
                        "stop": 1.0,
                        "speaker_id": "speaker1",
                        "ssml_gender": "male",
                        "translated_text": "Bonjour",
                    },
                    {
                        "text": "World",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker2",
                        "ssml_gender": "female",
                        "translated_text": "Le Monde",
                    },
                ],
            ),
        ],
    )
    def test_add_translations(
        self, utterance_metadata, translated_script, expected_translated_metadata
    ):
        updated_metadata = TranslationUT()._add_translations(
            utterance_metadata=utterance_metadata,
            translated_script=translated_script,
        )
        assert updated_metadata == expected_translated_metadata
