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
    analysis: dict
    
    def to_dict(self):
        return {
            "text": self.text,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "pause": self.pause,
            "analysis": self.analysis
        }

class ProsodySynthesizer:
    def __init__(self,
                 macro_alpha=0.2,
                 macro_pitch_shift_multiplier=1,
                 macro_speed_shift_multiplier=0.1,
                 macro_rate_responsiveness=0.1,
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
        base_rate = 0.45
        signed_squared_compound = (compound ** 2) if compound >= 0 else -(compound **2)
        rate_effect = pow(1.4142135, (signed_squared_compound * 0.4))
        energy_effect = (energy / 400)
        sentence_rate = max(0.1, min(2, (base_rate * rate_effect) + energy_effect))
        
        default_pbas = 1
        sentence_pbas = (default_pbas * pow(2, signed_squared_compound))
        sentence_pbas = max(0.08, min(4, sentence_pbas))

        base_pause = 0.2
        pause_effect = - (compound ** 2) if compound >= 0 else (abs(compound) ** 2) 
        sentence_pause = min(2, base_pause + pause_effect - (energy / 10))

        base_vol = 0.5
        max_boost = 0.5
        vol_boost = max_boost * pow(2, compound) * (energy / 10)
        vol_factor = min(1, base_vol + vol_boost * self.MACRO_VOLUME_RESPONSIVENESS)
        
        sentence_analysis = {
            "compound": compound,
            "energy": energy
        }
        
        return sentence_rate, sentence_pbas, vol_factor, sentence_pause, sentence_analysis

    def process_sentence(self, sentence, alpha=None) -> List[SpeechCommand]:
        """Process a sentence and return a list of SpeechCommands."""
        if alpha is None:
            alpha = self.MACRO_ALPHA
            
        compound, energy = self.analyze_sentiment_and_energy(sentence)
        sentence_rate, sentence_pitch, sentence_volume, sentence_pause, sentence_analysis = self.compute_sentence_parameters(compound, energy)
        command = [SpeechCommand(
                                text=sentence.casefold(),
                                rate=sentence_rate,
                                pitch=sentence_pitch,
                                volume=sentence_volume,
                                pause=sentence_pause,
                                analysis=sentence_analysis
                            )]
        return command

    def process_text(self, text) -> List[SpeechCommand]:
        """Process full text and return a list of SpeechCommands."""
        sentences = nltk.sent_tokenize(text)
        commands = []
        for sentence in sentences:
            commands.extend(self.process_sentence(sentence))
        return commands
    
    def get_speech_command(self, text) -> str:
        """Return JSON string of speech commands."""
        commands = self.process_text(text)
        return json.dumps([cmd.to_dict() for cmd in commands])