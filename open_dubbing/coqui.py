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
import re
import subprocess

from TTS.api import TTS


class Coqui:
    """Builds a list models available per each language"""

    def __init__(self, device="cpu"):
        logging.getLogger("TTS.utils.manage").setLevel(logging.ERROR)
        logging.getLogger("TTS.utils.audio.processor").setLevel(logging.ERROR)
        language_models = self._build_list_language_model()
        self.language_model = self._select_model_per_language(language_models)
        self.device = device

    @property
    def languages_model(self):
        return self.language_model

    def get_languages(self):
        return self.language_model.keys()

    """ Select first vits, then glow and then the rest"""

    def _select_model_per_language(self, language_models):
        language_model = {}
        for language in language_models:
            models = language_models[language]
            found = False
            if len(models) == 1:
                language_model[language] = models[0]
                found = True
            else:
                for model in models:
                    if "vits" in model:
                        language_model[language] = model
                        found = True
                        break

                for model in models:
                    if "glow" in model:
                        language_model[language] = model
                        found = True
                        break

            if found:
                continue

            language_model[language] = models[0]

        return language_model

    def debug_list_all_voices(self):
        for model in TTS.list_models():
            voices = TTS(model).speakers
            print(f"Model {model}: {voices}")

    def _build_list_language_model(self):
        language_models = {}

        for model in TTS.list_models():
            match = re.search(r"/([a-z]{2})/", model)
            if match:
                language = match.group(1)
                models = language_models.get(language, [])
                models.append(model)
                language_models[language] = models

        return language_models

    def get_voices_language(self, language):
        model = self.language_model[language]
        voices = TTS(model).speakers
        #      print(f"voices: {voices}")
        return voices

    def synthesize_speech(
        self, input_text, language, file_path, voice=None, audio_config=None
    ):
        model = self.language_model[language]
        tts = TTS(model).to(self.device)
        tts.tts_to_file(
            text=input_text, speaker=voice, split_sentences=False, file_path=file_path
        )

    @staticmethod
    def is_espeak_ng_installed():
        for cmd in [["espeak-ng", "--version"], ["espeak", "--version"]]:
            try:
                if (
                    subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    ).returncode
                    == 0
                ):
                    return True
            except FileNotFoundError:
                continue
        return False
