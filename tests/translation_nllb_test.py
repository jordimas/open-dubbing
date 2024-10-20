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

from unittest.mock import MagicMock, patch

from open_dubbing.translation_nllb import TranslationNLLB


class TestTranslationNLLB:

    def test_get_nllb_language(self):
        with patch(
            "open_dubbing.translation_nllb.TranslationNLLB._get_tokenizer_nllb"
        ) as mock_get_tokenizer_nllb:
            mock_tokenizer = MagicMock()
            mock_tokenizer.additional_special_tokens = [
                "cat_Latn",
                "ukr_Cyrl",
                "eng_Latn",
            ]
            mock_get_tokenizer_nllb.return_value = mock_tokenizer

            translation = TranslationNLLB()
            translation.tokenizer = mock_tokenizer

            assert translation._get_nllb_language("eng") == "eng_Latn"
            assert translation._get_nllb_language("ukr") == "ukr_Cyrl"
            assert translation._get_nllb_language("cat") == "cat_Latn"

    def test_get_language_pairs(self):
        with patch(
            "open_dubbing.translation_nllb.TranslationNLLB._get_tokenizer_nllb"
        ) as mock_get_tokenizer_nllb:
            mock_tokenizer = MagicMock()
            # Mock the return of lang_code_to_id.keys() to return a list of language codes
            mock_tokenizer.additional_special_tokens = [
                "cat_Latn",
                "eng_Latn",
                "fra_Latn",
            ]
            mock_get_tokenizer_nllb.return_value = mock_tokenizer

            translation = TranslationNLLB()
            translation.tokenizer = mock_tokenizer
            pairs = translation.get_language_pairs()

        expected_pairs = {
            ("cat", "eng"),
            ("cat", "fra"),
            ("eng", "cat"),
            ("eng", "fra"),
            ("fra", "cat"),
            ("fra", "eng"),
        }

        assert len(pairs) == 6
        assert pairs == expected_pairs
