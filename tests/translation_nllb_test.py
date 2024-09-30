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


from open_dubbing.translation_nllb import TranslationNLLB


class TestTranslationNLLB:

    def test_get_nllb_language(self):
        translation = TranslationNLLB()
        translation.load_model()

        assert translation._get_nllb_language("eng") == "eng_Latn"
        assert translation._get_nllb_language("ukr") == "ukr_Cyrl"
        assert translation._get_nllb_language("cat") == "cat_Latn"
