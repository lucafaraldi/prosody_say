import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import subprocess
import re
import numpy as np
import spacy

# Download required resources.
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model.
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Global constant for key parts-of-speech.
KEY_SET = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}

###############################################
# Macro Control Parameters (typical range 0–1)
###############################################
MACRO_ALPHA = 0.2                # Controls how much the local rate deviates from global.
MACRO_PITCH_SHIFT_MULTIPLIER = 0.5 # Dampens the computed pitch shift.
MACRO_SPEED_SHIFT_MULTIPLIER = 0.5 # Dampens the computed speed shift.
MACRO_RATE_RESPONSIVENESS = 0.8    # Scales the local rate adjustment.
MACRO_VOLUME_RESPONSIVENESS = 0.8  # Scales the global volume boost factor.

###############################################
# Global Parameter Computation
###############################################

def analyze_sentiment_and_energy(sentence):
    """
    Compute a continuous sentiment (compound) score and an energy value.
    Energy is derived from the standard deviation of word lengths plus a count
    of emphasis-worthy parts-of-speech (JJ, RB, VB, UH).
    """
    sentiment = sia.polarity_scores(sentence)
    compound = sentiment['compound']
    
    words = nltk.word_tokenize(sentence)
    word_lengths = [len(word) for word in words if word.isalpha()]
    lexical_variability = np.std(word_lengths) if word_lengths else 0
    
    pos_tags = nltk.pos_tag(words)
    emphasis_score = sum(1 for word, tag in pos_tags if tag in {"JJ", "RB", "VB", "UH"})
    
    energy = lexical_variability + emphasis_score
    return compound, energy

def compute_global_parameters(compound, energy):
    """
    Compute global (sentence-level) parameters:
      - global_rate: overall speech rate (WPM)
      - global_pbas: overall pitch baseline
      - global_pmod: overall pitch modulation (applied globally)
      - global_pause: pause duration at sentence end (ms)
      - vol_factor: volume boost factor for key words
    These serve as baselines for local adjustments.
    """
    # Global Speech Rate.
    base_rate = 140
    rate_effect = (compound ** 2) * 30 if compound >= 0 else - (abs(compound) ** 2) * 30
    energy_effect = (energy / 10) * 20
    global_rate = int(max(110, min(190, base_rate + rate_effect + energy_effect)))
    
    # Global Pause Duration.
    base_pause = 600
    pause_effect = - (compound ** 2) * 150 if compound >= 0 else (abs(compound) ** 2) * 150
    global_pause = max(200, int(base_pause + pause_effect + (energy / 10) * 100))
    
    # Global Pitch Baseline.
    default_pbas = 50
    global_pbas = (default_pbas + (compound ** 2) * 10) if compound >= 0 else (default_pbas - (abs(compound) ** 2) * 10)
    global_pbas = max(40, min(60, global_pbas))
    
    # Global Pitch Modulation (pmod): computed nonlinearly from energy.
    default_pmod = 20
    normalized_energy = energy / 10  # assumed range [0,10]
    global_pmod = default_pmod + (normalized_energy ** 2) * 10
    global_pmod = max(20, min(50, global_pmod))
    
    # Global Volume Boost Factor.
    base_vol = 1.0
    max_boost = 0.5
    if compound >= 0:
        vol_boost = max_boost * (compound ** 2) * (energy / 10)
        vol_factor = base_vol + vol_boost * MACRO_VOLUME_RESPONSIVENESS
    else:
        vol_boost = max_boost * (abs(compound) ** 2) * (energy / 10)
        vol_factor = base_vol - vol_boost * MACRO_VOLUME_RESPONSIVENESS
    
    return global_rate, global_pbas, global_pmod, global_pause, vol_factor

###############################################
# Dynamic Shift Computation (Macro-Controlled)
###############################################

def compute_dynamic_shifts(energy, compound, base_pitch_shift=5, base_speed_shift=10):
    """
    Compute dynamic shift values for pitch and speed based on energy and the
    absolute value of the compound sentiment.
    The computed shifts are scaled by macro multipliers so that higher energy and stronger sentiment produce larger shifts,
    but are damped by the macro parameters.
    """
    normalized_energy = energy / 10.0  # Normalize energy (assumed range 0-10)
    pitch_shift = base_pitch_shift * MACRO_PITCH_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
    speed_shift = base_speed_shift * MACRO_SPEED_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
    return pitch_shift, speed_shift

###############################################
# Local (Phrase and Word-Level) Prosody Processing (No Grouping)
###############################################

def apply_dynamic_prosody_to_phrase_no_group(phrase_tokens, compound, baseline_rate, global_pbas, pitch_shift, speed_shift):
    """
    Process a phrase (list of spaCy tokens) and output a flat string with dynamic per-word prosody,
    without grouping tokens together.
    
    For each non-punctuation token, compute a dynamic pitch value and a dynamic rate value,
    using linear interpolation over the phrase. For punctuation tokens, output them as is.
    
    Returns a flat string with inline commands, with each word having its own [[rate ...]] and [[pbas ...]] command.
    """
    # Get indices for non-punctuation tokens.
    word_indices = [i for i, token in enumerate(phrase_tokens) if not token.is_punct]
    n = len(word_indices)
    pitch_map = {}
    rate_map = {}
    if n > 0:
        if compound >= 0:
            start_pitch = global_pbas - pitch_shift
            end_pitch = global_pbas + pitch_shift
        else:
            start_pitch = global_pbas + pitch_shift
            end_pitch = global_pbas - pitch_shift
        if compound >= 0:
            start_rate = baseline_rate - speed_shift
            end_rate = baseline_rate + speed_shift
        else:
            start_rate = baseline_rate + speed_shift
            end_rate = baseline_rate - speed_shift
        for j, idx in enumerate(word_indices):
            if n > 1:
                pitch_val = start_pitch + (end_pitch - start_pitch) * (j / (n - 1))
                rate_val = start_rate + (end_rate - start_rate) * (j / (n - 1))
            else:
                pitch_val = start_pitch
                rate_val = start_rate
            pitch_map[idx] = int(round(pitch_val))
            rate_map[idx] = int(round(rate_val))
    else:
        pitch_map = {}
        rate_map = {}
    
    output_parts = []
    for i, token in enumerate(phrase_tokens):
        if token.is_punct:
            output_parts.append(token.text)
        else:
            token_out = f"[[rate {rate_map[i]}]][[pbas {pitch_map[i]}]]{token.text}"
            output_parts.append(token_out)
    
    return " ".join(output_parts)

###############################################
# Sentence-Level Processing
###############################################

def process_sentence(sentence, alpha=MACRO_ALPHA, base_pitch_shift=5, base_speed_shift=10):
    """
    Process a sentence by:
      1. Computing global parameters from the entire sentence.
      2. Computing dynamic shift values (pitch and speed) based on energy and sentiment, scaled by macros.
      3. Splitting the sentence into phrases (using punctuation boundaries).
      4. For each phrase:
            - Compute a local baseline rate relative to the global rate (using local key-word density).
            - Apply dynamic prosody (per-word rate and pitch adjustments) using the local baseline and computed dynamic shifts.
      5. Prepend global commands (global pbas, global pmod, global rate) to the entire sentence.
      6. Append a sentence-level pause.
    Global parameters serve as baselines; local adjustments are computed as relative deviations.
    """
    compound, energy = analyze_sentiment_and_energy(sentence)
    global_rate, global_pbas, global_pmod, global_pause, vol_factor = compute_global_parameters(compound, energy)
    
    # Compute dynamic shifts from energy and |compound|.
    pitch_shift, speed_shift = compute_dynamic_shifts(energy, compound)
    
    # Process the sentence using spaCy.
    doc = nlp(sentence)
    
    # Compute global importance: fraction of non-punctuation tokens that are key words.
    non_punct = [token for token in doc if not token.is_punct]
    global_importance = (sum(1 for token in non_punct if token.pos_ in KEY_SET) / len(non_punct)) if non_punct else 0
    
    # Split the sentence into phrases at punctuation boundaries.
    phrases = []
    current_phrase = []
    punctuation_set = {",", ";", ":", "-", ".", "!", "?"}
    for token in doc:
        current_phrase.append(token)
        if token.text in punctuation_set:
            phrases.append(current_phrase)
            current_phrase = []
    if current_phrase:
        phrases.append(current_phrase)
    
    phrase_outputs = []
    for phrase_tokens in phrases:
        # Compute local importance for the phrase.
        phrase_words = [t for t in phrase_tokens if not t.is_punct]
        local_importance = (sum(1 for t in phrase_words if t.pos_ in KEY_SET) / len(phrase_words)) if phrase_words else 0
        
        # Compute a local baseline rate for the phrase relative to the global rate.
        local_rate = int(global_rate * (1 - MACRO_RATE_RESPONSIVENESS * alpha * (local_importance - global_importance)))
        
        # Apply dynamic prosody to the phrase without grouping.
        phrase_str = apply_dynamic_prosody_to_phrase_no_group(phrase_tokens, compound, local_rate, global_pbas, pitch_shift, speed_shift)
        phrase_outputs.append(phrase_str)
    
    processed_local = " ".join(phrase_outputs)
    
    # Prepend global commands and append a sentence-level pause.
    processed_sentence = f"[[pbas {global_pbas}]][[pmod {global_pmod}]][[rate {global_rate}]]" + processed_local
    processed_sentence += f"[[slnc {global_pause}]]"
    return processed_sentence

def process_text(text):
    """
    Split the input text into sentences and process each sentence.
    Returns a list of processed sentences.
    """
    sentences = nltk.sent_tokenize(text)
    return [process_sentence(sentence) for sentence in sentences]

def speak_text(text):
    """
    Process the text and use macOS TTS (via the 'say' command) to speak it.
    """
    processed_sentences = process_text(text)
    for sentence in processed_sentences:
        print("Processed:", sentence)
        # Extract the global rate from the sentence for TTS.
        rate_match = re.search(r'\[\[rate (\d+)\]\]', sentence)
        rate = rate_match.group(1) if rate_match else "140"
        subprocess.run(["say", "-r", rate, sentence])

###############################################
# Example Usage
###############################################

if __name__ == "__main__":
    test_text = (
        "Losing you has left a void that can never be filled. "
        "Rest in peace. "
        "Your memory is a treasure I hold dear, and my heart aches with your absence. "
        "Gone too soon, but forever in my heart. "
        "I miss you more than words can say. "
        "The pain of your loss is overwhelming, but your spirit lives on in my memories. "
        "Each day without you is a reminder of how much you meant to me. "
        "Your absence leaves a heartache no one can heal, but your love leaves a memory no one can steal. "
        "I can't believe you're gone. "
        "My heart is broken, and my soul is aching. "
        "You may be gone, but your love and memories remain with me forever. "
        "Life without you is not the same. "
        "I miss you deeply. "
        "Your departure has left a hole in my heart that will never be filled."
    )
    speak_text(test_text)


# ###############################################
# # Example Usage
# ###############################################

# if __name__ == "__main__":
#     test_text = (
#         "Losing you has left a void that can never be filled. "
#         "Rest in peace. "
#         "Your memory is a treasure I hold dear, and my heart aches with your absence. "
#         "Gone too soon, but forever in my heart. "
#         "I miss you more than words can say. "
#         "The pain of your loss is overwhelming, but your spirit lives on in my memories. "
#         "Each day without you is a reminder of how much you meant to me. "
#         "Your absence leaves a heartache no one can heal, but your love leaves a memory no one can steal. "
#         "I can't believe you're gone. "
#         "My heart is broken, and my soul is aching. "
#         "You may be gone, but your love and memories remain with me forever. "
#         "Life without you is not the same. "
#         "I miss you deeply. "
#         "Your departure has left a hole in my heart that will never be filled."
#     )
    
#     synthesizer = ProsodySynthesizer()
#     # Get the speech command string.
#     command = synthesizer.get_speech_command(test_text)
#     print("Final Speech Command for say:\n", command)
#     # Optionally, speak it:
#     # synthesizer.speak_text(test_text)
