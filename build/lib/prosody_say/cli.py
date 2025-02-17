# prosody_say/cli.py
import nltk
import spacy.cli

def download_resources():
    """Download all required NLTK and spaCy resources."""
    print("Downloading NLTK resources...(vader_lexicon, punkt_tab, averaged_perceptron_tagger_eng)")
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    print("All resources downloaded successfully.")

if __name__ == '__main__':
    download_resources()
