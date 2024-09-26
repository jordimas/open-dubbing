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

import logging
from open_dubbing.text_to_speech import TextToSpeech
from typing import Mapping
import edge_tts
from edge_tts import VoicesManager, list_voices
from iso639 import Lang
import asyncio
import re


class TextToSpeechEdge(TextToSpeech):

    async def _create_manager(self):
        voice_manager = await VoicesManager.create()
        return voice_manager

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        results = {}
        iso_639_1 = self._get_iso_639_1(language_code)

        voice_manager = asyncio.run(self._create_manager())

        voices = voice_manager.find(Gender="Male", Language=iso_639_1)
        results["Male"] = voices[0]["ShortName"]
        voices = voice_manager.find(Gender="Female", Language=iso_639_1)
        results["Female"] = voices[0]["ShortName"]

        logging.debug(
            f"text_to_speech_edge.get_available_voices: {results} for language {language_code}"
        )

        return results

    def _get_iso_639_1(self, iso_639_3: str):
        o = Lang(iso_639_3)
        iso_639_1 = o.pt1
        return iso_639_1

    def _does_voice_supports_speeds(self):
        return True

    async def _save(self, text, speed, assigned_voice, output_filename):
        per = (100 * speed) - 100
        str_per = f"+{per:0.0f}%"
        communicate = edge_tts.Communicate(text, assigned_voice, rate=str_per)
        await communicate.save(output_filename)

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

        asyncio.run(self._save(text, speed, assigned_voice, output_filename))
        logging.debug(
            f"text_to_speech.client_edge.synthesize_speech: assigned_voice: {assigned_voice}, output_filename: '{output_filename}'"
        )
        return output_filename

    async def _get_list_voices(self):
        return await list_voices()

    def get_languages(self):
        voices = asyncio.run(self._get_list_voices())

        pattern = r"^([a-z]{2})-(.*)$"
        locales = set()
        for voice in voices:
            locale = voice["Locale"]
            _match = re.search(pattern, locale)
            if not _match:
                continue

            iso_639_1 = _match.group(1)
            o = Lang(iso_639_1)
            iso_639_3 = o.pt3

            locales.add(iso_639_3)
        languages = sorted(locales)
        logging.debug(f"text_to_speech.client_edge.get_languages: {languages}'")
        return languages
