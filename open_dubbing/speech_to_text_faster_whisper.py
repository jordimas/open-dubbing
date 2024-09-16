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

from faster_whisper import WhisperModel
import logging
from iso639 import Lang
from open_dubbing.speech_to_text import SpeechToText


class SpeechToTextFasterWhisper(SpeechToText):

    def __init__(self, device="cpu", cpu_threads=0):
        super().__init__(device, cpu_threads)
        logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    def load_model(self):
        self._model = WhisperModel(
            model_size_or_path="medium",
            device=self.device,
            cpu_threads=self.cpu_threads,
            compute_type="float16" if self.device == "cuda" else "int8",
        )

    def get_languages(self):
        iso_639_3 = []
        for language in self.model.supported_languages:
            if language == "jw":
                language = "jv"

            o = Lang(language)
            pt3 = o.pt3
            iso_639_3.append(pt3)
        return iso_639_3

    def _transcribe(
        self,
        *,
        vocals_filepath: str,
        source_language_iso_639_1: str,
    ) -> str:
        segments, _ = self.model.transcribe(
            vocals_filepath,
            source_language_iso_639_1,
        )
        return " ".join(segment.text for segment in segments)
