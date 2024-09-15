import tempfile
import os
import json
from faster_whisper import WhisperModel


class TestCmd:

    # TODO: To check transcription out of the final video
    def _get_transcription(self, filename):
        model = WhisperModel("medium")
        segments, info = model.transcribe(filename, language="ca", temperature=[0])
        text = ""
        for segment in segments:
            text += segment.text
        return text.strip(), info.language

    def test_translations_with_coqui(self):
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)

        _file = os.path.join(path, "englishvideo.mp4")
        with tempfile.TemporaryDirectory() as directory:
            command = (
                "open-dubbing "
                f"--input_file='{_file}' "
                f"--output_directory='{directory}' "
                "--source_language=eng "
                "--target_language=cat "
                "--tts=coqui"
            )
            cmd = f"cd {directory} && {command}"
            os.system(cmd)

            metadata_file = os.path.join(directory, "utterance_metadata_cat.json")
            with open(metadata_file) as json_data:
                data = json.load(json_data)
                text_array = [entry["translated_text"] for entry in data]
                assert "Bon dia, em dic Jordi Mas." == text_array[0]
                assert "Sóc de Barcelona." == text_array[1]
                assert "I m'encanta aquesta ciutat." == text_array[2]
