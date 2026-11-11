# EI Chatbot – Lightweight Response Refinement

A mini-project extending a sentiment-analysis chatbot with lightweight
interpretation of common slang, emoji cues, and possible sarcasm/irony.

## Features
- Sentiment analysis using VADER on the original user message.
- Normalization of a small dictionary of common chat abbreviations.
- Basic emoji cue interpretation.
- Conservative phrase/emoji-based possible-sarcasm flags.
- Responses adjusted using detected cues.
- Streamlit chat interface.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests
```bash
pytest
```

## Informal-language handling
- **Sarcasm / irony**: scored (0 to 0.95) from several weak signals: stock phrases,
  `/s`, positive wording about a negative situation, scare-quoted words,
  sarcasm-leaning emoji, plus ALL-CAPS / `!!` / elongation as boosters. Levels:
  `possible` (>= 0.5) and `likely` (>= 0.8). Negation and "but" clauses suppress
  false positives, and previous-turn sentiment is only ever a weak signal.
- **Slang**: abbreviations are expanded (`idk`, `tbh`, `gr8`...); modern slang
  (`mid`, `slay`, `no cap`, `wtf`...) carries a small sentiment polarity.
  Context-sensitive words (`sick`, `fire`, `goat`) only count in praise contexts.
- **Emoji**: ~40 emoji with meaning, polarity and repeat count. Variation
  selectors and skin tones are handled. Ambiguous emoji (😭 💀 🥲 ...) take their
  direction from the text ("so happy 😭" vs "so sad 😭").
- **Scoring**: `SentimentAnalyzer.analyze_with_context()` scores the text with
  emoji removed, then applies slang, emoji and sarcasm adjustments. The result
  keeps `raw_compound` and an `adjustments` list for transparency.
- **Responses**: sarcasm is checked before thanks/greetings, "hello, my order is
  late" is not treated as a bare greeting, and "thanks but..." gets an apology.

## Limitations
Sarcasm and emoji meanings are context-dependent. The current implementation
uses heuristics, not a trained sarcasm classifier. VADER is a lightweight
lexicon-based baseline and can still misread ambiguous informal messages.
Thresholds, slang polarities and emoji weights are hand-tuned starting points:
evaluate them on labelled data from your own users before relying on them.

## Attribution
This project extends
[Chatbot With Sentiment Analysis](https://github.com/Chesta31832/chatbot-sentiment-analyzer).
Retain the upstream MIT license and attribution when redistributing.
