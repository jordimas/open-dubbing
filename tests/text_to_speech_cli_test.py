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

import os

from open_dubbing.text_to_speech_cli import TextToSpeechCLI


class TestTextToSpeech:

    def test_get_available_voices(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        data_json = os.path.join(directory, "data/tts_cli.json")
        tts = TextToSpeechCLI(configuration_file=data_json)
        voices = tts.get_available_voices("cat")
        assert 2 == len(voices)
