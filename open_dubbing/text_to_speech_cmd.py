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
import os

from typing import Mapping

from open_dubbing.text_to_speech import TextToSpeech


class TextToSpeechCmd(TextToSpeech):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        voices = {}

        if language_code == "cat":
            voices["Male"] = "2"
            voices["Female"] = "3"

        logging.debug(
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

        directory = "/home/jordi/sc/open-dubbing2/Matcha-TTS"
        # python3 matcha_vocos_inference.py --output_path=/output/path --text_input="Bon dia Manel, avui anem a la muntanya." --length_scale=0.8 --temperature=0.7 --speaker_id 0 --cleaner "catalan_balear_cleaners"

        cmd = f'python3 matcha_vocos_inference.py --output_path=output/ --speaker_id {assigned_voice} --text_input="{text}"'
        cmd = f"cd {directory} && {cmd}"
        logging.info(f"cmd: {cmd}")
        os.system(cmd)
        wav_file = os.path.join(directory, f"output/spk_{assigned_voice}/synth.wav")
        self._convert_to_mp3(wav_file, output_filename)

        logging.debug(f"text_to_speech_cmd._convert_text_to_speech: {text}")

        return output_filename

    def get_languages(self):
        return ["cat"]
