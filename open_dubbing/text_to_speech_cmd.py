# Copyright 2024 Jordi Mas i Herǹadez <jmas@softcatala.org>
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
import logging
import os

from typing import Mapping

from open_dubbing.text_to_speech import TextToSpeech, Voice


class TextToSpeechCmd(TextToSpeech):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.configuration = self.load_json()

    def load_json(self):
        with open("/home/jordi/sc/open-dubbing2/samples/tts_sample.json", "r") as file:
            data = json.load(file)

        return data

    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        voices = []

        for voice_cfg in self.configuration["voices"]:
            if voice_cfg["language"] != language_code:
                continue

            voice = Voice(
                name=voice_cfg["id"],
                gender=voice_cfg["gender"],
                region=voice_cfg["region"],
            )
            voices.append(voice)

        logging.info(
            f"text_to_speech_cmd.get_available_voices: {voices} for language {language_code}"
        )
        return voices

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

        command = self.configuration["command"]
        cmd = command.format(assigned_voice=assigned_voice, text=text)
        logging.info(f"cmd: {cmd}")
        os.system(cmd)
        wav_file = os.path.join(directory, f"output/spk_{assigned_voice}/synth.wav")
        self._convert_to_mp3(wav_file, output_filename)

        logging.debug(f"text_to_speech_cmd._convert_text_to_speech: {text}")

        return output_filename

    def get_languages(self):
        languages = set()
        for voice_cfg in self.configuration["voices"]:
            language = voice_cfg["language"]
            languages.add(language)

        languages_list = list(languages)

        logging.info(f"text_to_speech_cmd.get_languages: {languages_list}")
        return languages_list
