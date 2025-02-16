# tests/test_prosody.py

import re
from unittest.mock import patch
from prosody_say import ProsodySynthesizer

def test_multisentence_command():
    text = ("hi, how are you doing? I am happy. I am sad. "
            "I am happy. I am sad. I am happy. I am sad. "
            "I am happy. I am sad.")

    synth = ProsodySynthesizer()
    command = synth.get_speech_command(text)

    # Ensure we got a non-empty command.
    assert command, "The TTS command should not be empty."

    # Check that there's at least one [[rate ...]] tag in the result.
    rate_match = re.search(r'\[\[rate (\d+)\]\]', command)
    assert rate_match, "Should contain a [[rate X]] command for at least one word."

    # Similarly, look for a pitch tag like [[pbas number]].
    pitch_match = re.search(r'\[\[pbas (\d+)\]\]', command)
    assert pitch_match, "Should contain a [[pbas Y]] command for at least one word."

def test_multisentence_speak():
    text = ("hi, how are you doing? I am happy. I am sad. "
            "I am happy. I am sad. I am happy. I am sad. "
            "I am happy. I am sad.")

    synth = ProsodySynthesizer()

    # Mock out subprocess.run so we don't actually call 'say'
    with patch("subprocess.run") as mock_run:
        synth.speak_text(text)
        # Check that it was called once
        mock_run.assert_called_once()
        # Inspect args
        called_args, called_kwargs = mock_run.call_args
        # The first arg should be ["say", "-r", <rate>, <full command>]
        assert called_args[0][0] == "say", "Should call the 'say' command."
        assert "-r" in called_args[0], "Should specify a rate with -r."
