# Introduction

Executing _open-dubbing  --help_ produces the following output:

```text
usage: open-dubbing [-h] --input_file INPUT_FILE [--output_directory OUTPUT_DIRECTORY] [--source_language SOURCE_LANGUAGE] --target_language
                    TARGET_LANGUAGE [--hugging_face_token HUGGING_FACE_TOKEN] [--tts {mms,coqui,edge,cli}] [--stt {auto,faster-whisper,transformers}]
                    [--translator {nllb,apertium}] [--apertium-server APERTIUM_SERVER] [--device {cpu,cuda}] [--cpu_threads CPU_THREADS] [--debug]
                    [--nllb_model {nllb-200-1.3B,nllb-200-3.3B}] [--whisper_model {medium,large-v3}] [--target_language_region TARGET_LANGUAGE_REGION]
                    [--tts_cli_cfg_file TTS_CLI_CFG_FILE]

AI dubbing system which uses machine learning models to automatically translate and synchronize audio dialogue into different languages

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Path to the input video file.
  --output_directory OUTPUT_DIRECTORY
                        Directory to save output files.
  --source_language SOURCE_LANGUAGE
                        Source language (ISO 639-3)
  --target_language TARGET_LANGUAGE
                        Target language for dubbing (ISO 639-3).
  --hugging_face_token HUGGING_FACE_TOKEN
                        Hugging Face API token.
  --tts {mms,coqui,edge,cli}
                        Text to Speech engine to use. Choices are:'mms': Meta Multilingual Speech engine, supports many languages.'coqui': Coqui TTS,
                        an open-source alternative for high-quality TTS.'edge': Microsoft Edge TSS.'cli': User defined TTS invoked from command line
  --stt {auto,faster-whisper,transformers}
                        Speech to text. Choices are:'auto': Autoselect best implementation.'faster-whisper': Faster-whisper's OpenAI whisper
                        implementation.'transformers': Transformers OpenAI whisper implementation.
  --translator {nllb,apertium}
                        Text to Speech engine to use. Choices are:'nllb': Meta's no Language Left Behind (NLLB).'apertium'': Apertium compatible API
                        server
  --apertium-server APERTIUM_SERVER
                        Apertium's URL server to use
  --device {cpu,cuda}   Device to use
  --cpu_threads CPU_THREADS
                        number of threads used for CPU inference (if is not specified uses defaults for each framework)
  --debug               keep intermediate files and generate specific files for debugging
  --nllb_model {nllb-200-1.3B,nllb-200-3.3B}
                        NLLB translation model size. 'nllb-200-3.3B' gives best translation quality
  --whisper_model {medium,large-v3}
                        name of the OpenAI Whisper speech to text model size to use
  --target_language_region TARGET_LANGUAGE_REGION
                        For some TTS you can specify the region of the language. For example, 'ES' will indicate accent from Spain.
  --tts_cli_cfg_file TTS_CLI_CFG_FILE
                        JSon configuration file when using a TTS which is involved by command line.


```

# How it works

The system follows these steps:

1. Isolate the speech from background noise, music, and other non-speech elements in the audio.
2. Segment the audio in fragments where there is voice and identify the speakers (speaker diarization).
3. Identify the gender of the speakers.
4. Transcribe the speech (STT) into text using OpenAI Whisper.
5. Translate the text from source language (e.g. English) to target language (e.g. Catalan).
6. Synthesize speech using a Text to Speech System (TTS) using voices that match the gender and adjusting speed.
7. The final dubbed video is then assembled, combining the synthetic audio with the original video footage, including any background sounds or music that were isolated earlier.

There are 6 different AI models applied during the dubbing process.

# Speech to text (SST)

For speech to text we use OpenAI Whisper. We provide two implementations:

* HuggingFace transformer's
* faster-whisper

faster-whisper works on Linux and it is a better implementation. HuggingFace transformer works in mac OS and Linux.

It is possible to add support for new Speech to text engines by extending the class _SpeechToText_

# TTS (text to speech)

Currently the system supports the following TTS systems:

- MMS: Meta Multilingual Speech engine, supports many languages
  - Pros
    - Supports over 1000 languages
  - Cons
    - Does not allow to select the voice (not possible to have male and female voices)
* Coqui TT
  - Pros
    - Possibility to add new languages
  - Cons
    - Many languages only support a single voice (not possible to have male and female voices)
* Microsoft Edge TSS served based
  - Pros
    - Good quality for the languages supported
  - Cons
    - This is a closed source option only for benchmarking
* CLI TTS
  * Allows you to use any TTS that can be called from the command line
    
The main driver to decide which TTS to use is the quality for your target language and the number of voices supported.

## Extending support for new TTS engines
    
It is possible to add support for new TTS engines by extending the class _TextToSpeech_

Additionally CLI TTS, allows you to use any TTS that can be called from the command line.

You need to provide a configuration file (see [tss_cli_sample.json](./samples/tss_cli_sample.json)
and call it like this.

```shell
 open-dubbing --input_file video.mp4 --tts="cmd" --tts_cmd_cfg_file="your_tts_configuration.json"
```

# Translation

We currently support two translation engines:

* Meta's [No Language Left Behind](https://ai.meta.com/research/no-language-left-behind/)
* [Apertium](https://www.apertium.org/) open source translation API

It is possible to add support for new TTS engines by extending the class _Translation_


