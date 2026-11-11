from typing import Any, Dict, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SentimentAnalyzer:
    """Analyzes sentiment of text using VADER, with optional adjustment for
    slang, emoji and possible sarcasm."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of given text

        Returns:
            dict: Contains 'neg', 'neu', 'pos', 'compound' scores
        """
        return self.analyzer.polarity_scores(text)

    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Convert compound score to readable label

        Args:
            compound_score: Score between -1 and 1

        Returns:
            str: 'Positive', 'Negative', or 'Neutral'
        """
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def analyze_with_label(self, text: str) -> Tuple[Dict[str, float], str]:
        """
        Analyze text and return scores with label

        Returns:
            tuple: (scores_dict, sentiment_label)
        """
        scores = self.analyze(text)
        label = self.get_sentiment_label(scores['compound'])
        return scores, label

    def analyze_with_context(
        self, text: str, interpretation: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Sentiment that accounts for informal-language cues.

        Steps:
          1. VADER scores the user's own wording with emoji removed. (VADER
             reads emoji by their literal names, e.g. "skull", which misreads
             slang uses such as 💀. Emoji are handled below instead. Slang is
             not expanded, so "tbh" doesn't inject sentiment words.)
          2. Slang polarity nudges the score (e.g. "mid" -, "slay" +).
          3. Emoji nudge the score. Ambiguous emoji (😭 💀 ...) amplify
             whichever direction the text already leans and add nothing
             to neutral text.
          4. If sarcasm is flagged, a positive score is flipped to negative
             (scaled by confidence) and a neutral score is made mildly negative.

        Returns (scores, label). scores['compound'] is the adjusted value;
        scores['raw_compound'] is VADER's untouched value and
        scores['adjustments'] lists what changed it.
        """
        interp = interpretation or {}
        clean_text = interp.get("text_no_emoji", text)
        scores: Dict[str, Any] = dict(self.analyze(clean_text))
        raw = scores["compound"]
        compound = raw
        adjustments = []

        # -- slang
        slang_nudge = _clamp(
            sum(item.get("polarity", 0.0) for item in interp.get("slang", [])),
            -0.6, 0.6,
        )
        if slang_nudge:
            compound += slang_nudge
            adjustments.append(f"slang {slang_nudge:+.2f}")

        # -- emoji
        emoji_nudge = 0.0
        for cue in interp.get("emoji_cues", []):
            repeats = min(cue.get("count", 1), 3)
            intensity = 1.0 + 0.25 * (repeats - 1)
            polarity = cue.get("polarity")
            if polarity is None:  # ambiguous: follow the text's direction
                if compound >= 0.05:
                    polarity = 0.2
                elif compound <= -0.05:
                    polarity = -0.2
                else:
                    polarity = 0.0
            emoji_nudge += polarity * intensity
        emoji_nudge = _clamp(emoji_nudge, -0.6, 0.6)
        if emoji_nudge:
            compound += emoji_nudge
            adjustments.append(f"emoji {emoji_nudge:+.2f}")
        compound = _clamp(compound)

        # -- sarcasm
        if interp.get("possible_sarcasm"):
            confidence = interp.get("sarcasm_score", 0.5)
            before = compound
            if compound >= 0.05:
                compound = -max(abs(compound) * confidence, 0.2)
            elif compound > -0.05:
                compound = -0.15
            if compound != before:
                adjustments.append(
                    f"sarcasm ({interp.get('sarcasm_level', 'possible')}): "
                    f"{before:+.2f} -> {compound:+.2f}"
                )

        scores["raw_compound"] = raw
        scores["compound"] = round(_clamp(compound), 4)
        scores["adjustments"] = adjustments
        return scores, self.get_sentiment_label(scores["compound"])
