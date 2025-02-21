import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import numpy as np
import spacy
from dataclasses import dataclass
from typing import List
# Download required resources (if not already present).

@dataclass
class SpeechCommand:
    text: str
    rate: float
    pitch: float
    volume: float
    pause: float
    
    def to_dict(self):
        return {
            "text": self.text,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "pause": self.pause
        }

class ProsodySynthesizer:
    def __init__(self,
                 macro_alpha=0.2,
                 macro_pitch_shift_multiplier=0.5,
                 macro_speed_shift_multiplier=10,
                 macro_rate_responsiveness=0.8,
                 macro_volume_responsiveness=0.8):
        self.MACRO_ALPHA = macro_alpha
        self.MACRO_PITCH_SHIFT_MULTIPLIER = macro_pitch_shift_multiplier
        self.MACRO_SPEED_SHIFT_MULTIPLIER = macro_speed_shift_multiplier
        self.MACRO_RATE_RESPONSIVENESS = macro_rate_responsiveness
        self.MACRO_VOLUME_RESPONSIVENESS = macro_volume_responsiveness

        self.sia = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.KEY_SET = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}
    
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
    
    def compute_sentence_parameters(self, compound:float, energy:float):
        """Compute sentence-level baseline parameters."""
        base_rate = 140
        signed_squared_compound= (compound ** 2) if compound>= 0 else -(compound **2)
        rate_effect = pow(1.4142135, (signed_squared_compound* 0.5))
        energy_effect = (energy / 10)
        sentence_rate = max(100, min(190, (base_rate * rate_effect) + energy_effect))
        
        default_pbas = 1
        sentence_pbas = (default_pbas * pow(2, signed_squared_compound))
        sentence_pbas = max(0.08, min(4, sentence_pbas))

        base_pause = 600
        pause_effect = - (compound ** 2) * 150 if compound >= 0 else (abs(compound) ** 2) * 150
        sentence_pause = max(200, int(base_pause + pause_effect + (energy / 10) * 100))

        base_vol = 0.5
        max_boost = 0.5
        vol_boost = max_boost * pow(2,compound ) * (energy / 10)
        vol_factor = min(1, base_vol+ vol_boost* self.MACRO_VOLUME_RESPONSIVENESS)

        return sentence_rate, sentence_pbas, vol_factor, sentence_pause
    
    def compute_dynamic_shifts(self, energy, compound):
        """Compute per-word shift amounts for pitch and speed."""
        normalized_energy = energy / 10.0
        pitch_shift = self.MACRO_PITCH_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
        speed_shift =  self.MACRO_SPEED_SHIFT_MULTIPLIER * (1 + normalized_energy * abs(compound))
        return pitch_shift, speed_shift

    def process_sentence(self, sentence, alpha=None) -> List[SpeechCommand]:
        """Process a sentence and return a list of SpeechCommands."""
        if alpha is None:
            alpha = self.MACRO_ALPHA
            
        compound, energy = self.analyze_sentiment_and_energy(sentence)
        sentence_rate, sentence_pbas, volume_factor, sentence_pause = self.compute_sentence_parameters(compound, energy)
        pitch_shift, speed_shift = self.compute_dynamic_shifts(energy, compound)
        
        doc = self.nlp(sentence)
        
        # Calculate importance factors
        non_punct = [token for token in doc if not token.is_punct]
        sentence_importance = (sum(1 for token in non_punct if token.pos_ in self.KEY_SET) / len(non_punct)) if non_punct else 0
        
        commands = []
        phrase_tokens = []
        # Group tokens into phrases
        for token in doc:
            if token.is_punct and token.text in {",", ";", ":", ".", "!", "?"}:
                if phrase_tokens:
                    # Process the accumulated phrase
                    phrase_words = [t for t in phrase_tokens if not t.is_punct]
                    local_importance = (sum(1 for t in phrase_words if t.pos_ in self.KEY_SET) / len(phrase_words)) if phrase_words else 0
                    local_rate = float(sentence_rate * (1 - self.MACRO_RATE_RESPONSIVENESS * alpha * (local_importance - sentence_importance)))
                    
                    # Process each word in the phrase
                    for i, word_token in enumerate(phrase_tokens):
                        
                        if not word_token.is_punct:  
                            progress = i / len(phrase_tokens)
                            direction= 1 if compound >= 0 else -1
                            word_pitch = sentence_pbas * pow(2, pitch_shift* progress* direction)
                            word_rate = local_rate + (speed_shift* progress* direction)
                            word_volume= volume_factor
                            commands.append(SpeechCommand(
                                text=word_token.text,
                                rate=word_rate,
                                pitch=word_pitch,
                                volume= word_volume,
                                pause= 0
                            ))
                        else:
                            commands[:-1].text= commands[:-1].text + word_token.text
                            commands[:-1].pause= sentence_pause
                phrase_tokens = []
            else:
                phrase_tokens.append(token)

        return commands

    def process_text(self, text) -> List[SpeechCommand]:
        """Process full text and return a list of SpeechCommands."""
        sentences = nltk.sent_tokenize(text)
        commands = []
        print(sentences)
        for sentence in sentences:
            commands.extend(self.process_sentence(sentence))
        return commands
    
    def get_speech_command(self, text) -> str:
        """Return JSON string of speech commands."""
        commands = self.process_text(text)
        return json.dumps([cmd.to_dict() for cmd in commands])

