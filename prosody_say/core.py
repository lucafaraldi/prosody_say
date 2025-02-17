import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import subprocess
import re
import numpy as np
import spacy

# Download required resources (if not already present).

class ProsodySynthesizer:
    def __init__(self,
                 macro_alpha=0.2,
                 macro_pitch_shift_multiplier=0.5,
                 macro_speed_shift_multiplier=0.5,
                 macro_rate_responsiveness=0.8,
                 macro_volume_responsiveness=0.8):
        # Macro parameters for controlling responsiveness.
        self.MACRO_ALPHA = macro_alpha
        self.MACRO_PITCH_SHIFT_MULTIPLIER = macro_pitch_shift_multiplier
        self.MACRO_SPEED_SHIFT_MULTIPLIER = macro_speed_shift_multiplier
        self.MACRO_RATE_RESPONSIVENESS = macro_rate_responsiveness
        self.MACRO_VOLUME_RESPONSIVENESS = macro_volume_responsiveness

        # We'll use NLTK's VADER for sentiment analysis.
        self.sia = SentimentIntensityAnalyzer()
        # Use spaCy for tokenization, POS tagging, and phrase splitting.
        self.nlp = spacy.load("en_core_web_sm")
        # Global constant for key parts-of-speech.
        self.KEY_SET = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}
        
    # --- Global Parameter Computation ---
    
    def analyze_sentiment_and_energy(self, sentence):
        """Compute a compound sentiment score and an energy value for a sentence."""
        sentiment = self.sia.polarity_scores(sentence)
        compound = sentiment['compound']
        
        words = nltk.word_tokenize(sentence)
        word_lengths = [len(word) for word in words if word.isalpha()]
        lexical_variability = np.std(word_lengths) if word_lengths else 0
        
        pos_tags = nltk.pos_tag(words)
        emphasis_score = sum(1 for word, tag in pos_tags if tag in {"JJ", "RB", "VB", "UH"})
        
        energy = lexical_variability + emphasis_score
        return compound, energy
    
    def compute_global_parameters(self, compound, energy):
        """Compute global baselines from the entire sentence."""
        base_rate = 140
        rate_effect = (compound ** 2) * 30 if compound >= 0 else - (abs(compound) ** 2) * 30
        energy_effect = (energy / 10) * 20
        global_rate = int(max(110, min(190, base_rate + rate_effect + energy_effect)))
        
        base_pause = 600
        pause_effect = - (compound ** 2) * 150 if compound >= 0 else (abs(compound) ** 2) * 150
        global_pause = max(200, int(base_pause + pause_effect + (energy / 10) * 100))
        
        default_pbas = 50
        global_pbas = (default_pbas + (compound ** 2) * 10) if compound >= 0 else (default_pbas - (abs(compound) ** 2) * 10)
        global_pbas = max(40, min(60, global_pbas))
        
        default_pmod = 20
        normalized_energy = energy / 10
        global_pmod = default_pmod + (normalized_energy ** 2) * 10
        global_pmod = max(20, min(50, global_pmod))
        
        base_vol = 1.0
        max_boost = 0.5
        if compound >= 0:
            vol_boost = max_boost * (compound ** 2) * (energy / 10)
            vol_factor = base_vol + vol_boost * self.MACRO_VOLUME_RESPONSIVENESS
        else:
            vol_boost = max_boost * (abs(compound) ** 2) * (energy / 10)
            vol_factor = base_vol - vol_boost * self.MACRO_VOLUME_RESPONSIVENESS
        
        return global_rate, global_pbas, global_pmod, global_pause, vol_factor
    
    # --- Dynamic Shift Computation ---
    
    def compute_dynamic_shifts(self, energy, compound, base_pitch_shift=5, base_speed_shift=10):
        """Compute per-word shift amounts for pitch and speed based on energy and sentiment."""
        normalized_energy = energy / 10.0
        pitch_shift = base_pitch_shift * self.MACRO_PITCH_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
        speed_shift = base_speed_shift * self.MACRO_SPEED_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
        return pitch_shift, speed_shift

    # --- Local (Per-Word) Prosody Processing without Grouping ---
    
    def apply_dynamic_prosody_to_phrase_no_group(self, phrase_tokens, compound, baseline_rate, global_pbas, pitch_shift, speed_shift):
        """
        Process a phrase (list of spaCy tokens) word-by-word without grouping.
        For each non-punctuation token, compute dynamic pitch and rate values via linear interpolation.
        Skip tokens that already look like inline commands to avoid reprocessing them.
        Return a flat string with inline commands.
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
            # If the token text is already an inline command, skip further processing.
            if token.text.startswith("[[") and token.text.endswith("]]"):
                output_parts.append(token.text)
            elif token.is_punct:
                output_parts.append(token.text)
            else:
                token_out = f"[[rate {rate_map[i]}]][[pbas {pitch_map[i]}]]{token.text}"
                output_parts.append(token_out)
        
        return " ".join(output_parts)

    
    # --- Sentence-Level Processing ---
    
    def process_sentence(self, sentence, alpha=None, base_pitch_shift=5, base_speed_shift=10):
        """
        Process a sentence:
         - Compute global parameters and dynamic shifts.
         - Split the sentence into phrases (based on punctuation).
         - For each phrase, compute a local baseline rate based on local key-word density.
         - Process each phrase word-by-word (no grouping) for dynamic prosody.
         - Prepend global commands and append a sentence-level pause.
        """
        if alpha is None:
            alpha = self.MACRO_ALPHA
        compound, energy = self.analyze_sentiment_and_energy(sentence)
        global_rate, global_pbas, global_pmod, global_pause, vol_factor = self.compute_global_parameters(compound, energy)
        pitch_shift, speed_shift = self.compute_dynamic_shifts(energy, compound)
        
        doc = self.nlp(sentence)
        
        non_punct = [token for token in doc if not token.is_punct]
        global_importance = (sum(1 for token in non_punct if token.pos_ in self.KEY_SET) / len(non_punct)) if non_punct else 0
        
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
            phrase_words = [t for t in phrase_tokens if not t.is_punct]
            local_importance = (sum(1 for t in phrase_words if t.pos_ in self.KEY_SET) / len(phrase_words)) if phrase_words else 0
            local_rate = int(global_rate * (1 - self.MACRO_RATE_RESPONSIVENESS * alpha * (local_importance - global_importance)))
            phrase_str = self.apply_dynamic_prosody_to_phrase_no_group(phrase_tokens, compound, local_rate, global_pbas, pitch_shift, speed_shift)
            phrase_outputs.append(phrase_str)
        
        processed_local = " ".join(phrase_outputs)
        processed_sentence = f"[[pbas {global_pbas}]][[pmod {global_pmod}]][[rate {global_rate}]]" + processed_local
        processed_sentence += f"[[slnc {global_pause}]]"
        return processed_sentence

    def process_text(self, text):
        """Split text into sentences and process each sentence."""
        sentences = nltk.sent_tokenize(text)
        return [self.process_sentence(sentence) for sentence in sentences]
    
    def get_speech_command(self, text):
        """Return the full processed speech command string for the given input text."""
        processed_sentences = self.process_text(text)
        # Join sentences with a space.
        return " ".join(processed_sentences)
    
    def speak_text(self, text):
        """Process the text and send it to macOS TTS via the 'say' command."""
        command = self.get_speech_command(text)
        # For the TTS command, we need a global rate. We'll extract it from the first sentence.
        rate_match = re.search(r'\[\[rate (\d+)\]\]', command)
        rate = rate_match.group(1) if rate_match else "140"
        print("Speech Command:", command)
        subprocess.run(["say", "-r", rate, command])

