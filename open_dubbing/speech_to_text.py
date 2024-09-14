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

from typing import Mapping, Sequence
from faster_whisper import WhisperModel
import logging
from iso639 import Lang


class SpeechToText:

    def __init__(self, device="cpu"):
        self.model = None
        self.device = device
        logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def load_model(self):
        self._model = WhisperModel(
            model_size_or_path="large-v3",
            device=self.device,
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

    def _get_iso_639_1(self, iso_639_3: str):
        o = Lang(iso_639_3)
        iso_639_1 = o.pt1
        return iso_639_1

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

    def transcribe_audio_chunks(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, float | str]],
        source_language: str,
        no_dubbing_phrases: Sequence[str],
    ) -> Sequence[Mapping[str, float | str]]:

        logging.debug(f"transcribe_audio_chunks: {source_language}")
        iso_639_1 = self._get_iso_639_1(source_language)

        updated_utterance_metadata = []
        for item in utterance_metadata:
            new_item = item.copy()
            transcribed_text = self._transcribe(
                vocals_filepath=item["path"],
                source_language_iso_639_1=iso_639_1,
            )
            dubbing = len(transcribed_text) > 0
            logging.debug(
                f"transcribe_audio_chunks. text: '{transcribed_text}' - dubbing: {dubbing}"
            )
            new_item["text"] = transcribed_text
            new_item["for_dubbing"] = dubbing
            updated_utterance_metadata.append(new_item)
        return updated_utterance_metadata

    # TODO: We need to gender prediciton here
    # This is later important for TTS
    def diarize_speakers(
        self,
        *,
        file: str,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        number_of_speakers: int,
    ) -> Sequence[tuple[str, str]]:

        r = []
        chunks = len(utterance_metadata)
        for chunk in range(0, chunks):
            _tuple = ("speaker_01", "Male")
            r.append(_tuple)

        logging.info(f"text_to_speech.diarize_speakers. Returns: {r}")
        return r

    def add_speaker_info(
        self,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        speaker_info: Sequence[tuple[str, str]],
    ) -> Sequence[Mapping[str, str | float]]:
        if len(utterance_metadata) != len(speaker_info):
            raise Exception(
                "The length of 'utterance_metadata' and 'speaker_info' must be the"
                " same."
            )
        updated_utterance_metadata = []
        for utterance, (speaker_id, ssml_gender) in zip(
            utterance_metadata, speaker_info
        ):
            new_utterance = utterance.copy()
            new_utterance["speaker_id"] = speaker_id
            new_utterance["ssml_gender"] = ssml_gender
            updated_utterance_metadata.append(new_utterance)
        return updated_utterance_metadata
