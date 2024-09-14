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

import re
from typing import Final, Mapping, Sequence
import logging
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import time


_BREAK_MARKER: Final[str] = "<BREAK>"


class Translation:

    def __init__(self, device="cpu"):
        self.model_name = "facebook/nllb-200-3.3B"
        self.device = device

    def generate_script(self, *, utterance_metadata, key: str = "text") -> str:
        """Generates a script string from a list of utterance metadata."""
        trimmed_lines = [
            item[key].strip() if item[key] else "" for item in utterance_metadata
        ]
        logging.info(f"translation.generate_script. Input: {trimmed_lines}")
        r = _BREAK_MARKER + _BREAK_MARKER.join(trimmed_lines) + _BREAK_MARKER
        logging.debug(f"translation.generate_script. Returns: {r}")
        return r

    def _get_tokenizer_nllb(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def _get_model_nllb(self):
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                self.device
            )
        except RuntimeError as e:
            if self.device == "cuda":
                return AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to("cpu")
                logging.warning(
                    f"Loading translation model {self.model_name} in CPU since cannot be load in GPU"
                )
            else:
                raise e

    def get_languages(self):
        tokenizer = self._get_tokenizer_nllb()
        # Returns 'cat_Latn'
        original_list = tokenizer.lang_code_to_id.keys()
        # Get only the language codes
        supported_languages = [s[:3] for s in original_list]
        return supported_languages

    def _process_and_translate(self, translator, input_string):
        # Split the input string by the <BREAK> delimiter
        parts = input_string.split(_BREAK_MARKER)

        translated_parts = []
        for part in parts:
            if len(part.strip()) == 0:
                translation = ""
            else:
                translated = translator(part)
                translation = translated[0]["translation_text"]

            translated_parts.append(translation)

        result = _BREAK_MARKER.join(translated_parts)
        return result

    def translate_script(
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
            f"translation.translate_script. Input source_language: {source_language}"
        )
        logging.debug(
            f"translation.translate_script. Input target_language: {target_language}"
        )

        tokenizer = self._get_tokenizer_nllb()
        model = self._get_model_nllb()
        source_language = source_language

        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=f"{source_language}" + "_Latn",  # TODO: Fix for other langs
            tgt_lang=f"{target_language}" + "_Latn",
            max_length=1024,
        )

        translation = self._process_and_translate(translator, script)
        logging.debug(f"translation.translate_script. Translation: {translation}")

        pretty_data = json.dumps(translation, indent=4, ensure_ascii=False)
        logging.debug(f"translation.translate_script. Returns: {pretty_data}")
        execution_time = time.time() - start_time
        logging.debug(
            "translation.translate_script. Time: %.2f seconds.", execution_time
        )

        return translation

    def add_translations(
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
