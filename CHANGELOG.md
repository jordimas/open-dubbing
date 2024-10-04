# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.0.5]

### Added

- Only speed audios when it's really needed improving quality of final audio syntesis
- Support for Apertium API as translation engine
- Allow to select between different model sizes for NLLB translation engine
- Allow to select between different model sizes for Whisper speech to text engine

## [0.0.4]

### Added

- Check if ffmpeg is installed and report if it is not

## [0.0.3]

### Added

- Autodetect language using Whisper if source language is not specified
- Use Edge TTS native speed parameter when need to increase the speed
- Better performance when separating vocals

## [0.0.2]

### Added

- Support for Microsoft Edge TTS
- Gender classifier to identify gender in the original video and produce the syntetic voices in target language that match the gender

## [0.0.1]

### Added
- Initial version
