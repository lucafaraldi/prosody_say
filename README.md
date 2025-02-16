# prosody_synthesizer

A prosody processing library for TTS synthesis on **macOS** that computes dynamic pitch, rate, and volume adjustments based on sentiment and energy analysis. This library relies on the system `say` command, which is available only on macOS.

---

## Features

- **Sentiment and Energy Analysis**: Uses NLTK’s VADER and basic lexical heuristics to derive a sentence‐level “energy” and sentiment (compound) score.
- **Dynamic Shifts**: Adjusts speech rate and pitch dynamically within each phrase.
- **macOS `say` Integration**: Outputs text with inline TTS commands (e.g. `[[rate]]`, `[[pbas]]`), then invokes `say`.
- **Object-Oriented Design**: Encapsulated in a `ProsodySynthesizer` class for easy usage.

---

## Requirements

1. **macOS**  
   This library uses the built-in `say` command, so it won't work on Windows or Linux unless you adapt those TTS calls.

2. **Python <3.12** (recommended)

   - Although no strict upper-bound is enforced, note that **spaCy** may not yet support Python 3.13.
   - If you encounter issues on future Python versions, consider using 3.6–3.12.

3. **Dependencies**
   - `nltk`
   - `numpy`
   - `spacy`

## Required Data Downloads

## Download Required Resources

Before using `prosody_say`, ensure that the necessary NLTK and spaCy resources are installed. You can do this by running:

```bash
prosody-download-resources
```

## Installation

1. **Install from PyPI** :
   ```bash
   pip install prosody_synthesizer
   ```

````

2. **Install from source**:
   ```git clone https://github.com/yourusername/prosody_synthesizer.git
   cd prosody_synthesizer
   pip install .
   ```
````
