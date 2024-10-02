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
import logging
import re
import time

from abc import ABC, abstractmethod
from typing import Final, Mapping, Sequence

_BREAK_MARKER: Final[str] = "<BREAK>"


class Translation(ABC):

    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def load_model(self):
        pass

    def _generate_script(self, *, utterance_metadata, key: str = "text") -> str:
        """Generates a script string from a list of utterance metadata."""
        trimmed_lines = [
            item[key].strip() if item[key] else "" for item in utterance_metadata
        ]
        logging.debug(f"translation.generate_script. Input: {trimmed_lines}")
        r = _BREAK_MARKER + _BREAK_MARKER.join(trimmed_lines) + _BREAK_MARKER
        logging.debug(f"translation.generate_script. Returns: {r}")
        return r

    @abstractmethod
    def get_language_pairs(self):
        pass

    @abstractmethod
    def _translate_text(
        self, source_language: str, target_language: str, text: str
    ) -> str:
        pass

    def translate_utterances(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        source_language: str,
        target_language: str,
    ) -> str:

        script = self._generate_script(utterance_metadata=utterance_metadata)
        translated_script = self._translate_script(
            script=script,
            source_language=source_language,
            target_language=target_language,
        )
        return self._add_translations(
            utterance_metadata=utterance_metadata,
            translated_script=translated_script,
        )

    def _translate_script(
        self,
        *,
        script: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translates the provided transcript to the target language.
        Input: <BREAK>Hello, my name is Jordi Mas.<BREAK>I'm from Barcelona<BREAK>and I also work in Barcelona.<BREAK>Thanks a lot for listening.<BREAK>For.<BREAK>
        Returns <BREAK>Hello, my name is Jordi Mas.<BREAK>I'm from Barcelona<BREAK>and I also work in Barcelona.<BREAK>Thanks a lot for listening.<BREAK>For.<BREAK>

        """

        start_time = time.time()

        logging.debug(f"translation.translate_script. Input script: {script}")
        logging.debug(
            f"translation.translate_script. Input source_language: {source_language}, target_language: {target_language}"
        )

        # Split the input string by the <BREAK> delimiter
        parts = script.split(_BREAK_MARKER)

        translated_parts = []
        for text in parts:
            if len(text.strip()) == 0:
                translation = ""
            else:
                translation = self._translate_text(
                    source_language=source_language,
                    target_language=target_language,
                    text=text,
                )

            translated_parts.append(translation)

        translation = _BREAK_MARKER.join(translated_parts)
        logging.debug(f"translation.translate_script. Translation: {translation}")

        pretty_data = json.dumps(translation, indent=4, ensure_ascii=False)
        logging.debug(f"translation.translate_script. Returns: {pretty_data}")
        execution_time = time.time() - start_time
        logging.debug(
            "translation.translate_script. Time: %.2f seconds.", execution_time
        )

        return translation

    def _add_translations(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        translated_script: str,
    ) -> Sequence[Mapping[str, str | float]]:
        """Updates the "translated_text" field of each utterance metadata with the corresponding text segment."""
        stripped_translation = re.sub(
            rf"^\s*{_BREAK_MARKER}\s*|\s*{_BREAK_MARKER}\s*$", "", translated_script
        )
        text_segments = stripped_translation.split(_BREAK_MARKER)
        if len(utterance_metadata) != len(text_segments):
            raise ValueError(
                "The utterance metadata must be of the same length as the text"
                f" segments. Currently they are: {len(utterance_metadata)} and"
                f" {len(text_segments)}."
            )
        updated_utterance_metadata = []
        for metadata, translated_text in zip(utterance_metadata, text_segments):
            updated_utterance_metadata.append(
                {**metadata, "translated_text": translated_text}
            )

        pretty_data = json.dumps(
            updated_utterance_metadata, indent=4, ensure_ascii=False
        )
        logging.debug(f"translation.translated_script. Input: {translated_script}")
        logging.debug(f"translation.translated_script. Returns: {pretty_data}")
        return updated_utterance_metadata
