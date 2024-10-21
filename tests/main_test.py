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

from unittest.mock import patch

import pytest

import open_dubbing

from open_dubbing.main import _get_selected_tts


class TestMain:

    def test_get_selected_tts_mss(self):
        tts = open_dubbing.main._get_selected_tts("mms", "", "cpu")
        assert "TextToSpeechMMS" == type(tts).__name__

    def test_get_selected_tts_edge(self):
        tts = _get_selected_tts("edge", "", "cpu")
        assert "TextToSpeechEdge" == type(tts).__name__

    def test_get_selected_tts_cli(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        data_json = os.path.join(directory, "data/tts_cli.json")
        tts = _get_selected_tts("cli", data_json, "cpu")
        assert "TextToSpeechCLI" == type(tts).__name__

    def test_get_selected_tts_cli_no_cfg_file(self):
        with pytest.raises(SystemExit) as excinfo:
            _get_selected_tts("cli", "", "cpu")

        assert excinfo.type == SystemExit
        assert excinfo.value.code == 108
