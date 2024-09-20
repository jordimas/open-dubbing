# Copyright 2024 Jordi Mas i Herǹandez <jmas@softcatala.org>
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
from open_dubbing.coqui import Coqui
from iso639 import Lang
from typing import Mapping


class TextToSpeechCoqui(TextToSpeech):

    def __init__(self, device="cpu"):
        super().__init__()
        self.coqui = Coqui(device)

    def get_languages(self):
        languages = []
        for iso_639_1 in self.coqui.get_languages():
            o = Lang(iso_639_1)
            iso_639_3 = o.pt3
            languages.append(iso_639_3)

        return languages

    def _get_iso_639_1(self, iso_639_3: str):
        o = Lang(iso_639_3)
        iso_639_1 = o.pt1
        return iso_639_1

    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        voices = {}

        if language_code == "cat":
            voices["Male"] = "pau"
            voices["Female"] = "ona"

        logging.debug(
            f"text_to_speech_coqui.get_available_voices: {voices} for language {language_code}"
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

        wav_file = output_filename.replace(".mp3", ".wav")
        logging.debug(
            f"text_to_speech.client.synthesize_speech: pre synthesize_speech: '{text}', '{target_language}', file: {wav_file}, speed: {speed}, voice: {assigned_voice}"
        )
        iso_639_1 = self._get_iso_639_1(target_language)
        self.coqui.synthesize_speech(
            text, iso_639_1, file_path=wav_file, voice=assigned_voice
        )

        self._convert_to_mp3(wav_file, output_filename)
        logging.debug(
            f"text_to_speech.client.synthesize_speech: output_filename: '{output_filename}'"
        )
        return output_filename
