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

import json

from unittest.mock import MagicMock, patch

from open_dubbing.translation_apertium import TranslationApertium


class TestTranslationApertium:
    def test_translate_text(self):
        translation_apertium = TranslationApertium()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            expected_response = {"responseData": {"translatedText": "Hola món"}}
            mock_response.read.return_value = json.dumps(expected_response).encode(
                "utf-8"
            )
            mock_urlopen.return_value = mock_response

            # Test data
            source_language = "eng"
            target_language = "cat"
            text = "Hello World"
            translation_apertium.set_server("http://fake-server/")

            translated_text = translation_apertium._translate_text(
                source_language, target_language, text
            )

            # Verify the correct URL is generated and check the translation result
            mock_urlopen.assert_called_once_with(
                "http://fake-server/translate?q=Hello+World&langpair=eng|cat&markUnknown=no"
            )
            assert translated_text == "Hola món"

    def test_get_language_pairs(self):

        translation_apertium = TranslationApertium()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            expected_response = {
                "responseData": [
                    {"sourceLanguage": "eng", "targetLanguage": "cat"},
                    {"sourceLanguage": "cat", "targetLanguage": "fra"},
                ]
            }
            mock_response.read.return_value = json.dumps(expected_response).encode(
                "utf-8"
            )
            mock_urlopen.return_value = mock_response

            # Set the server and get the language pairs
            translation_apertium.set_server("http://fake-server/")
            language_pairs = translation_apertium.get_language_pairs()

            # Check that the correct URL is generated and assert the results
            mock_urlopen.assert_called_once_with("http://fake-server/listPairs")
            assert language_pairs == {("eng", "cat"), ("cat", "fra")}
