[![PyPI version](https://img.shields.io/pypi/v/open-dubbing.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/open-dubbing/)
[![PyPI downloads](https://img.shields.io/pypi/dm/open-dubbing.svg)](https://pypistats.org/packages/open-dubbing)

# Introduction

Open dubbing is an AI dubbing system uses machine learning models to automatically translate and synchronize audio dialogue into different languages.

At the moment, it is pure *experimental* and an excuse to help me to understand better STT, TTS and translation systems combined together.

# Features

* Build on top of open source models and able to run it locally
* Dubs automatically a video from a source to a target language
* Supports multiple Text To Speech (TTS) engines
* Gender voice detection to allow to assign properly synthetic voice

# Roadmap

Areas what we will like to explore:

* Automatic detection of the source language of the video (using Whisper)
* Better control of voice used for dubbing
* Support for TTS systems
* Optimize it for long videos and less resource usage
* Support for multiple video input formats

# Demo

This video on propose shows the strengths and limitations of the system.

*Original English video*

https://github.com/user-attachments/assets/54c0d37f-0cc8-4ea2-8f8d-fd2d2f4eeccc

*Automatic dubbed video in Catalan*


https://github.com/user-attachments/assets/99936655-5851-4d0c-827b-f36f79f56190


# Limitations

* This is an experimental project
* Automatic video dubbing includes speech recognition, translation, vocal recognition, etc. At each one of these steps errors can be introduced

# Supported languages

The support languages depends on the combination of text to speech, translation system and text to speech system used. With Coqui TTS, these are the languages supported (I only tested a very few of them):

Supported source languages: Afrikaans, Amharic, Armenian, Assamese, Bashkir, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Burmese, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Faroese, Finnish, French, Galician, Georgian, German, Gujarati, Haitian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Lao, Lingala, Lithuanian, Luxembourgish, Macedonian, Malayalam, Maltese, Maori, Marathi, Modern Greek (1453-), Norwegian Nynorsk, Occitan (post 1500), Panjabi, Polish, Portuguese, Romanian, Russian, Sanskrit, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish, Turkmen, Ukrainian, Urdu, Vietnamese, Welsh, Yoruba, Yue Chinese

Supported target languages: Achinese, Akan, Amharic, Assamese, Awadhi, Ayacucho Quechua, Balinese, Bambara, Bashkir, Basque, Bemba (Zambia), Bengali, Bulgarian, Burmese, Catalan, Cebuano, Central Aymara, Chhattisgarhi, Crimean Tatar, Dutch, Dyula, Dzongkha, English, Ewe, Faroese, Fijian, Finnish, Fon, French, Ganda, German, Guarani, Gujarati, Haitian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Iloko, Indonesian, Javanese, Kabiyè, Kabyle, Kachin, Kannada, Kazakh, Khmer, Kikuyu, Kinyarwanda, Kirghiz, Korean, Lao, Magahi, Maithili, Malayalam, Marathi, Minangkabau, Modern Greek (1453-), Mossi, North Azerbaijani, Northern Kurdish, Nuer, Nyanja, Odia, Pangasinan, Panjabi, Papiamento, Polish, Portuguese, Romanian, Rundi, Russian, Samoan, Sango, Shan, Shona, Somali, South Azerbaijani, Southwestern Dinka, Spanish, Sundanese, Swahili (individual language), Swedish, Tagalog, Tajik, Tamasheq, Tamil, Tatar, Telugu, Thai, Tibetan, Tigrinya, Tok Pisin, Tsonga, Turkish, Turkmen, Uighur, Ukrainian, Urdu, Vietnamese, Waray (Philippines), Welsh, Yoruba


# Installation

## Install dependencies

Linux:

```shell
sudo apt install ffmpeg
```
Mac OS
```shell
brew install ffmpeg
```

If you are going to use Coqui-tts you also need to install espeak-ng:

```shell
sudo apt install espeak-ng
```
Mac OS
```shell
brew install espeak-ng
```

Install package:

```shell
pip install open_dubbing
```

## Accept pyannote license

1. Go to and Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
2. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
3. Go to and access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

# Usage

Quick start

```shell

 open-dubbing  --input_file video.mp4 --target_language=cat --hugging_face_token=TOKEN
```
Where _TOKEN_ is the HuggingFace token that allows to access the models

To get a list of available options:

```shell
open-dubbing --help
```

# Libraries used

Core libraries used:
* [demucs](https://github.com/facebookresearch/demucs) to separate vocals from the audio
* [pyannote-audio](https://github.com/pyannote/pyannote-audio) to diarize speakers
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for audio to speech
* [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) for machine translation
* TTS
  * [coqui-tts](https://github.com/idiap/coqui-ai-TTS)
  * Meta [mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
  * Microsoft [Edge TTS](https://github.com/rany2/edge-tts)

And very special thanks to [ariel](https://github.com/google-marketing-solutions/ariel) from which we leveraged parts of their code base.

# License

See [license](./LICENSE)

# How it works

The system follows these steps:

1. Isolate the speech from background noise, music, and other non-speech elements in the audio.
2. Segment the audio in fragments where there is voice and identify the speakers (speaker diarization).
3. Identify the gender of the speakers.
4. Transcribe the speech into text using OpenAI Whisper.
5. Translate the text from source language (e.g. English) to target language (e.g. Catalan).
6. Synthesize speech using a Text to Speech System using voices that match the gender and adjusting speed.
7. The final dubbed video is then assembled, combining the synthetic audio with the original video footage, including any background sounds or music that were isolated earlier.

There are 6 different AI models applied during the dubbing process.

# Contact

Email address: Jordi Mas: jmas@softcatala.org
