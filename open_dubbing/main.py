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

import argparse

import sys
import os
from open_dubbing.dubbing import Dubber
import logging
from open_dubbing.speech_to_text_faster_whisper import SpeechToTextFasterWhisper
from open_dubbing.speech_to_text_whisper_transformers import (
    SpeechToTextWhisperTransfomers,
)
from open_dubbing.translation import Translation
from open_dubbing.text_to_speech_mms import TextToSpeechMMS
from open_dubbing.text_to_speech_coqui import TextToSpeechCoqui
from open_dubbing.text_to_speech_edge import TextToSpeechEdge
from iso639 import Lang
from open_dubbing.coqui import Coqui


def _init_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the global log level

    # File handler for logging to a file
    file_handler = logging.FileHandler("open_dubbing.log")
    file_handler.setLevel(logging.DEBUG)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def check_languages(source_language, target_language, _tts, _sst):
    spt = _sst.get_languages()
    trans = Translation().get_languages()
    tts = _tts.get_languages()

    if source_language not in spt:
        raise ValueError(
            f"source language '{source_language}' is not supported by the speech recognition system. Supported languages: '{spt}"
        )

    if source_language not in trans:
        raise ValueError(
            f"source language '{source_language}' is not supported by the translation system. Supported languages: '{trans}"
        )

    if target_language not in trans:
        raise ValueError(
            f"target language '{target_language}' is not supported by the translation system. Supported languages: '{trans}"
        )

    if target_language not in tts:
        raise ValueError(
            f"target language '{target_language}' is not supported by the text to speech system. Supported languages: '{tts}"
        )


_ACCEPTED_VIDEO_FORMATS = ["mp4"]


def check_is_a_video(input_file: str):
    _, file_extension = os.path.splitext(input_file)
    file_extension = file_extension.lower().lstrip(".")

    if file_extension in _ACCEPTED_VIDEO_FORMATS:
        return
    raise ValueError(f"Unsupported file format: {file_extension}")


HUGGING_FACE_VARNAME = "HF_TOKEN"


def get_token(provided_token: str) -> str:
    token = provided_token or os.getenv(HUGGING_FACE_VARNAME)
    if not token:
        raise ValueError(
            f"You must either provide the '--hugging_face_token' argument or"
            f" set the '{HUGGING_FACE_VARNAME.upper()}' environment variable."
        )
    return token


def _get_language_names(languages_iso_639_3):
    names = []
    for language in languages_iso_639_3:
        o = Lang(language)
        names.append(o.name)
    return sorted(names)


def list_supported_languages(_tts, device):  # TODO: Not used
    s = SpeechToTextFasterWhisper(device=device)
    s.load_model()
    spt = s.get_languages()
    trans = Translation().get_languages()
    tts = _tts.get_languages()

    source = _get_language_names(set(spt).intersection(set(trans)))
    print(f"Supported source languages: {source}")

    target = _get_language_names(set(tts).intersection(set(trans)))
    print(f"Supported target languages: {target}")


def main():
    _init_logging()
    """Parses command-line arguments and runs the dubbing process."""
    parser = argparse.ArgumentParser(description="Run the end-to-end dubbing process.")
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input video or audio file.",
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--source_language",
        required=True,
        help="Source language (ISO 639-3)",
    )
    parser.add_argument(
        "--target_language",
        required=True,
        help="Target language for dubbing (ISO 639-3).",
    )
    parser.add_argument(
        "--hugging_face_token",
        default=None,
        help="Hugging Face API token.",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default="mms",
        choices=["mms", "coqui", "edge"],
        help=(
            "Text to Speech engine to use. Choices are:"
            "'mms': Meta Multilingual Speech engine, supports many languages."
            "'coqui': Coqui TTS, an open-source alternative for high-quality TTS."
            "'edge': Microsoft Edge TSS."
        ),
    )
    parser.add_argument(
        "--stt",
        type=str,
        default="auto",
        choices=["auto", "faster-whisper", "transformers"],
        help=(
            "Speech to text. Choices are:"
            "'auto': Autoselect best implementation."
            "'faster-whisper': Faster-whisper's OpenAI whisper implementation."
            "'transformers': Transformers OpenAI whisper implementation."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=("Device to use"),
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="number of threads used for CPU inference (if not specified uses defaults for each framework)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="keep intemediate files and generate specific files for debugging",
    )

    args = parser.parse_args()

    check_is_a_video(args.input_file)

    hugging_face_token = get_token(args.hugging_face_token)

    if args.tts == "mms":
        tts = TextToSpeechMMS(args.device)
    elif args.tts == "edge":
        tts = TextToSpeechEdge(args.device)
    elif args.tts == "coqui":
        tts = TextToSpeechCoqui(args.device)
        if not Coqui.is_espeak_ng_installed():
            raise ValueError(
                "To use Coqui-tts you have to have espeak or espeak-ng installed"
            )
    else:
        raise ValueError(f"Invalid tts value {args.tts}")

    if sys.platform == "darwin":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.stt == "auto":
        if sys.platform == "darwin":
            stt = SpeechToTextWhisperTransfomers(args.device, args.cpu_threads)
        else:
            stt = SpeechToTextFasterWhisper(args.device, args.cpu_threads)
    elif args.stt == "faster-whisper":
        stt = SpeechToTextFasterWhisper(args.device, args.cpu_threads)
    else:
        stt = SpeechToTextWhisperTransfomers(args.device, args.cpu_threads)

    stt.load_model()
    check_languages(args.source_language, args.target_language, tts, stt)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    dubber = Dubber(
        input_file=args.input_file,
        output_directory=args.output_directory,
        source_language=args.source_language,
        target_language=args.target_language,
        hugging_face_token=hugging_face_token,
        tts=tts,
        stt=stt,
        device=args.device,
        cpu_threads=args.cpu_threads,
        debug=args.debug,
    )
    logging.info(
        f"Processing '{args.input_file}' file with tts '{args.tts}', sst {args.stt} and device '{args.device}'"
    )
    dubber.dub()


if __name__ == "__main__":
    main()
