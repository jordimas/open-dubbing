# Copyright 2024 Google LLC
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

from abc import ABC, abstractmethod
from typing import Final, Mapping, Sequence

from pydub import AudioSegment
from pydub.effects import speedup

_DEFAULT_CHUNK_SIZE: Final[int] = 150


class TextToSpeech(ABC):

    def __init__(self):
        self._SSML_MALE: Final[str] = "Male"
        self._SSML_FEMALE: Final[str] = "Female"
        self._DEFAULT_SPEED: Final[float] = 1.0
        self._DEFAULT_VOLUME_GAIN_DB: Final[float] = 16.0

    @abstractmethod
    def get_available_voices(self, language_code: str) -> Mapping[str, str]:
        pass

    def assign_voices(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        target_language: str,
        preferred_voices: Sequence[str] = "",
    ) -> Mapping[str, str | None]:

        voices = self.get_available_voices(target_language)
        voice_assignment = {}
        if len(voices) == 0:
            voice_assignment["speaker_01"] = "ona"
        else:
            for chunk in utterance_metadata:
                speaker_id = chunk["speaker_id"]
                if speaker_id in voice_assignment:
                    continue

                gender = chunk["ssml_gender"]
                voice = voices[gender]
                voice_assignment[speaker_id] = voice

        logging.debug(f"text_to_speech.assign_voices. Returns: {voice_assignment}")
        return voice_assignment

    def _convert_to_mp3(self, input_file, output_mp3):
        cmd = f"ffmpeg -y -i {input_file} {output_mp3} > /dev/null 2>&1"
        logging.debug(cmd)
        os.system(cmd)
        os.remove(input_file)

    def _add_text_to_speech_properties(
        self,
        *,
        utterance_metadata: Mapping[str, str | float],
    ) -> Mapping[str, str | float]:
        """Updates utterance metadata with Text-To-Speech properties."""
        utterance_metadata_copy = utterance_metadata.copy()
        voice_properties = dict(
            pitch=0,
            speed=self._DEFAULT_SPEED,
            volume_gain_db=self._DEFAULT_VOLUME_GAIN_DB,
        )
        utterance_metadata_copy.update(voice_properties)
        return utterance_metadata_copy

    def update_utterance_metadata(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        assigned_voices: Mapping[str, str] | None,
    ) -> Sequence[Mapping[str, str | float]]:
        """Updates utterance metadata with assigned voices."""
        updated_utterance_metadata = []
        for metadata_item in utterance_metadata:
            new_utterance = metadata_item.copy()
            speaker_id = new_utterance.get("speaker_id")
            new_utterance["assigned_voice"] = assigned_voices.get(speaker_id)
            new_utterance = self._add_text_to_speech_properties(
                utterance_metadata=new_utterance
            )
            updated_utterance_metadata.append(new_utterance)
        return updated_utterance_metadata

    @abstractmethod
    def get_languages(self):
        pass

    @abstractmethod
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
        pass

    def _calculate_target_utterance_speed(
        self,
        *,
        reference_length: float,
        dubbed_file: str,
    ) -> float:
        """Returns the ratio between the reference and target duration."""

        dubbed_audio = AudioSegment.from_file(dubbed_file)
        dubbed_duration = dubbed_audio.duration_seconds
        r = dubbed_duration / reference_length
        logging.debug(f"text_to_speech._calculate_target_utterance_speed: {r}")
        return r

    def create_speaker_to_paths_mapping(
        self,
        utterance_metadata: Sequence[Mapping[str, float | str]],
    ) -> Mapping[str, Sequence[str]]:
        """Organizes a list of utterance metadata dictionaries into a speaker-to-paths mapping.

        Returns:
            A mapping between speaker IDs to lists of file paths.
        """

        speaker_to_paths_mapping = {}
        for utterance in utterance_metadata:
            speaker_id = utterance["speaker_id"]
            if speaker_id not in speaker_to_paths_mapping:
                speaker_to_paths_mapping[speaker_id] = []
            speaker_to_paths_mapping[speaker_id].append(utterance["vocals_path"])
        return speaker_to_paths_mapping

    def _adjust_audio_speed(
        self,
        *,
        reference_length: float,
        dubbed_file: str,
        speed: float,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Adjusts the speed of an MP3 file to match the reference file duration.

        The speed will not be adjusted if the dubbed file has a duration that
        is the same or shorter than the duration of the reference file.
        """

        dubbed_audio = AudioSegment.from_file(dubbed_file)
        logging.info(
            "Adjusting audio speed will prevent overlaps of utterances. However,"
            " it might change the voice sligthly."
        )
        crossfade = max(1, chunk_size // 2)

        logging.debug(
            f"text_to_speech.adjust_audio_speed: dubbed_audio: {dubbed_file}, speed: {speed}, chunk_size: {chunk_size}, crossfade: {crossfade}"
        )
        output_audio = speedup(
            dubbed_audio, speed, chunk_size=chunk_size, crossfade=crossfade
        )
        output_audio.export(dubbed_file, format="mp3")

    def _does_voice_supports_speeds(self):
        return False

    def dub_utterances(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        output_directory: str,
        target_language: str,
        adjust_speed: bool = True,
    ) -> Sequence[Mapping[str, str | float]]:
        """Processes a list of utterance metadata, generating dubbed audio files."""

        logging.debug(f"TextToSpeech.dub_utterances: adjust_speed: {adjust_speed}")
        updated_utterance_metadata = []
        for utterance in utterance_metadata:
            utterance_copy = utterance.copy()
            if not utterance_copy["for_dubbing"]:
                try:
                    dubbed_path = utterance_copy["path"]
                except KeyError:
                    dubbed_path = f"chunk_{utterance['start']}_{utterance['end']}.mp3"
            else:
                assigned_voice = utterance_copy["assigned_voice"]
                reference_length = utterance_copy["end"] - utterance_copy["start"]
                text = utterance_copy["translated_text"]
                try:
                    path = utterance_copy["path"]
                    base_filename = os.path.splitext(os.path.basename(path))[0]
                    output_filename = os.path.join(
                        output_directory, f"dubbed_{base_filename}.mp3"
                    )
                except KeyError:
                    output_filename = os.path.join(
                        output_directory,
                        f"dubbed_chunk_{utterance['start']}_{utterance['end']}.mp3",
                    )

                speed = utterance_copy["speed"]
                dubbed_path = self._convert_text_to_speech(
                    assigned_voice=assigned_voice,
                    target_language=target_language,
                    output_filename=output_filename,
                    text=text,
                    pitch=utterance_copy["pitch"],
                    speed=speed,
                    volume_gain_db=utterance_copy["volume_gain_db"],
                )
                assigned_voice = utterance_copy.get("assigned_voice", None)
                assigned_voice = assigned_voice if assigned_voice else ""
                support_speeds = self._does_voice_supports_speeds()
                speed = self._calculate_target_utterance_speed(
                    reference_length=reference_length, dubbed_file=dubbed_path
                )
                logging.debug(f"support_speeds: {support_speeds}, speed: {speed}")
                if speed > 1.0:
                    MAX_SPEED = 1.3
                    if speed > MAX_SPEED:
                        logging.debug(
                            f"text_to_speech.dub_utterances: Reduced speed from {speed} to {MAX_SPEED}"
                        )
                        speed = MAX_SPEED

                    utterance_copy["speed"] = speed
                    if support_speeds:
                        dubbed_path = self._convert_text_to_speech(
                            assigned_voice=assigned_voice,
                            target_language=target_language,
                            output_filename=output_filename,
                            text=text,
                            pitch=utterance_copy["pitch"],
                            speed=speed,
                            volume_gain_db=utterance_copy["volume_gain_db"],
                        )
                    else:
                        chunk_size = utterance_copy.get(
                            "chunk_size", _DEFAULT_CHUNK_SIZE
                        )
                        self._adjust_audio_speed(
                            reference_length=reference_length,
                            dubbed_file=dubbed_path,
                            speed=speed,
                            chunk_size=chunk_size,
                        )
                        utterance_copy["chunk_size"] = chunk_size

            utterance_copy["dubbed_path"] = dubbed_path
            updated_utterance_metadata.append(utterance_copy)
        return updated_utterance_metadata
