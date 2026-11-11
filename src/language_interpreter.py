"""
Lightweight interpretation of informal chat language.

Detects:
  * abbreviations / slang (normalised and/or given a small sentiment polarity)
  * emoji (with counts, polarity and context-dependent meanings)
  * possible sarcasm / irony, scored from several weak signals

Everything here is heuristic. It produces *cues with a confidence*, not
definitive classifications, and it deliberately prefers false negatives over
false positives for sarcasm.
"""
import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
# A term is only matched when not glued to other word characters / hyphens,
# and not followed by ".x" (so "U.S." is not read as the slang word "u").
_L = r"(?<![\w-])"
_R = r"(?![\w-]|\.\w)"


def _term_re(term: str) -> "re.Pattern":
    return re.compile(_L + re.escape(term) + _R, re.IGNORECASE)


# Emoji modifiers: variation selector, zero-width joiner, skin tones.
_MODIFIERS = re.compile("[\ufe0f\u200d\U0001F3FB-\U0001F3FF]")
# Pictographic characters (used to strip emoji before lexicon-based scoring).
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF\u2600-\u27BF\u2B50\u2B55]"
)

# Marks an emoji whose polarity depends on the surrounding text.
AMBIGUOUS = None


def strip_emoji(text: str) -> str:
    """Remove emoji/pictographs (used so a text lexicon never sees them)."""
    text = _MODIFIERS.sub("", text)
    return re.sub(r"\s{2,}", " ", _EMOJI_RE.sub("", text)).strip()


class LanguageInterpreter:
    # ----------------------------------------------------------------------
    # Abbreviations: safe to expand in the normalised text.
    # (Single letters such as "w", "l", "r" were removed: they corrupted
    # ordinary text like "an L on my exam".)
    # ----------------------------------------------------------------------
    SLANG_MAP = {
        "idk": "I do not know",
        "idc": "I do not care",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tbh": "to be honest",
        "tbf": "to be fair",
        "ngl": "not going to lie",
        "rn": "right now",
        "brb": "be right back",
        "btw": "by the way",
        "fyi": "for your information",
        "nvm": "never mind",
        "np": "no problem",
        "smh": "shaking my head",
        "ikr": "I know right",
        "omg": "oh my god",
        "lol": "laughing out loud",
        "lmao": "laughing hard",
        "rofl": "rolling on the floor laughing",
        "bc": "because",
        "cuz": "because",
        "tho": "though",
        "u": "you",
        "ur": "your",
        "pls": "please",
        "plz": "please",
        "thx": "thanks",
        "ty": "thank you",
        "fr": "for real",
        "gr8": "great",
        "asap": "as soon as possible",
        "tmrw": "tomorrow",
    }

    # ----------------------------------------------------------------------
    # Modern slang: reported, and given a small sentiment nudge (polarity)
    # because a plain lexicon (e.g. VADER) usually cannot read these.
    # term/regex -> (meaning, polarity)
    # Context-sensitive words (sick, fire, goat) use regexes so that
    # "I feel sick" / "fire alarm" / "a goat" are NOT read as praise.
    # ----------------------------------------------------------------------
    SLANG_TERMS = [
        (_term_re("lowkey"), "lowkey", "somewhat / secretly", 0.0),
        (_term_re("highkey"), "highkey", "very / openly", 0.0),
        (_term_re("no cap"), "no cap", "no lie / seriously", 0.0),
        (_term_re("cap"), "cap", "a lie (when said alone)", 0.0),
        (_term_re("bruh"), "bruh", "exasperation or disbelief", -0.1),
        (_term_re("bro"), "bro", "casual address", 0.0),
        (_term_re("af"), "af", "very (intensifier)", 0.0),
        (_term_re("mid"), "mid", "mediocre / disappointing", -0.4),
        (_term_re("slay"), "slay", "doing great / impressive", 0.4),
        (_term_re("slaps"), "slaps", "is really good", 0.4),
        (_term_re("banger"), "banger", "something excellent", 0.4),
        (_term_re("lit"), "lit", "exciting / great", 0.3),
        (_term_re("goated"), "goated", "the best", 0.5),
        (_term_re("cringe"), "cringe", "embarrassing / awkward", -0.4),
        (_term_re("sus"), "sus", "suspicious", -0.3),
        (_term_re("wtf"), "wtf", "frustration or shock", -0.3),
        (_term_re("ffs"), "ffs", "exasperation", -0.4),
        (_term_re("fml"), "fml", "things are going badly", -0.5),
        (_term_re("yikes"), "yikes", "alarm or discomfort", -0.3),
        (_term_re("ugh"), "ugh", "annoyance", -0.3),
        (_term_re("bet"), "bet", "agreement / okay", 0.1),
        (
            re.compile(r"(?:\bis|\bare|'s|\bwas)\s+(?:the\s+)?goat\b", re.I),
            "goat",
            "greatest of all time",
            0.5,
        ),
        (
            re.compile(
                r"(?:(?:this|that|it|its|it's|that's|is|so|lowkey|pure|straight)\s+fire\b)",
                re.I,
            ),
            "fire",
            "excellent / impressive",
            0.4,
        ),
        (
            re.compile(
                r"(?:(?:that's|thats|this is|it's|so|pretty|too|real|sick)\s+sick\b|\bsick\s+(?:bro|dude|man|work|job)\b)",
                re.I,
            ),
            "sick",
            "awesome (in this context)",
            0.4,
        ),
    ]

    # ----------------------------------------------------------------------
    # Emoji: char -> (meaning, polarity).  polarity AMBIGUOUS (None) means
    # the sign is taken from the surrounding text (e.g. "so happy 😭").
    # ----------------------------------------------------------------------
    EMOJI_CUES = {
        "😂": ("amusement", 0.3),
        "🤣": ("amusement", 0.3),
        "😭": ("strong emotion: sadness or overwhelmed joy; context decides", AMBIGUOUS),
        "💀": ("'I'm dead': amusement or disbelief; context decides", AMBIGUOUS),
        "🥲": ("bittersweet / holding back tears", AMBIGUOUS),
        "😅": ("nervous or awkward laugh", AMBIGUOUS),
        "🫠": ("overwhelmed or melting with embarrassment", AMBIGUOUS),
        "🤯": ("mind blown: awe or shock", AMBIGUOUS),
        "🙄": ("annoyance or possible sarcasm", -0.3),
        "🙃": ("possible irony or awkwardness", -0.1),
        "😒": ("annoyance", -0.3),
        "😑": ("unimpressed / deadpan", -0.2),
        "😐": ("neutral / unimpressed", -0.05),
        "🤦": ("facepalm: exasperation", -0.3),
        "😤": ("frustration", -0.3),
        "😡": ("anger", -0.5),
        "🤬": ("strong anger", -0.5),
        "😩": ("exasperation or weariness", -0.3),
        "😫": ("exhaustion or frustration", -0.3),
        "😔": ("sadness", -0.3),
        "😞": ("disappointment", -0.3),
        "😢": ("sadness", -0.3),
        "😬": ("awkwardness or discomfort", -0.1),
        "😱": ("shock or alarm", -0.2),
        "🤡": ("mockery / feeling foolish", -0.2),
        "👎": ("disapproval", -0.3),
        "❤️": ("affection or appreciation", 0.4),
        "😍": ("admiration", 0.4),
        "🥰": ("affection", 0.4),
        "🥹": ("touched / moved", 0.2),
        "😊": ("warm happiness", 0.3),
        "😀": ("happiness", 0.3),
        "😄": ("happiness", 0.3),
        "😁": ("happiness", 0.3),
        "😎": ("confidence / approval", 0.2),
        "🥳": ("celebration", 0.4),
        "🎉": ("celebration", 0.3),
        "🔥": ("enthusiasm or approval", 0.3),
        "💯": ("strong agreement / approval", 0.3),
        "👍": ("agreement or acknowledgement", 0.2),
        "👏": ("applause / approval", 0.3),
        "🙏": ("thanks or pleading", 0.0),
        "🙂": ("context-dependent; can be polite or ironic", 0.0),
        "🤔": ("uncertainty / thinking", 0.0),
    }
    # Text emoticons that VADER does not already handle.
    EMOTICONS = {"-_-": ("annoyance / deadpan", -0.2)}

    # ----------------------------------------------------------------------
    # Sarcasm signals
    # ----------------------------------------------------------------------
    SARCASM_PHRASES = (
        "yeah right",
        "yeah, right",
        "just what i needed",
        "love that for me",
        "what a surprise",
        "big surprise",
        "thanks for nothing",
        "thanks a lot for",
        "oh fantastic",
        "oh great",
        "oh wonderful",
        "oh perfect",
        "oh joy",
        "just great",
        "just perfect",
        "sure, whatever",
        "oh sure",
        "as if",
        "how nice",
        "real helpful",
        "so helpful",
        "nice job breaking",
        "great job breaking",
        "good job breaking",
        "because that totally",
        "this is fine",
        "way to go",
    )

    POSITIVE_WORDS = {
        "love", "loving", "great", "awesome", "fantastic", "wonderful",
        "perfect", "amazing", "brilliant", "nice", "thrilled", "helpful",
        "fun", "happy", "excellent", "lovely", "super", "genius", "good",
        "terrific", "delighted", "impressive",
    }
    NEGATIVE_SITUATION = re.compile(
        r"^(crash\w*|broke\w*|break\w*|wait\w*|again|ignor\w*|late|delay\w*|"
        r"stuck|error\w*|fail\w*|lost|useless|slow\w*|ruin\w*|bug\w*|wrong|"
        r"froze\w*|freez\w*|glitch\w*|unresponsive)$"
    )
    NEGATORS = {"not", "no", "never", "without", "dont", "didnt", "doesnt",
                "isnt", "wasnt", "wont", "cant", "couldnt", "wouldnt",
                "hasnt", "havent", "nothing", "n't"}
    CONTRAST = {"but", "however", "though", "although", "yet"}

    SARCASM_THRESHOLD = 0.5   # "possible"
    SARCASM_LIKELY = 0.8      # "likely"

    # ----------------------------------------------------------------------
    def interpret(
        self, text: str, previous_sentiment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpret informal-language cues in `text`.

        previous_sentiment: label ("Positive"/"Negative"/"Neutral") of the
        user's previous message, used as weak context for sarcasm.
        """
        original = text
        cleaned_modifiers = _MODIFIERS.sub("", original)

        slang, normalized = self._find_slang(original)
        emoji_cues = self._find_emoji(cleaned_modifiers)
        sarcasm = self._score_sarcasm(
            original, cleaned_modifiers, emoji_cues, previous_sentiment
        )
        tone = self._tone_cues(original)

        return {
            "original_text": original,
            "normalized_text": normalized,
            "text_no_emoji": strip_emoji(original),
            "slang": slang,
            "emoji_cues": emoji_cues,
            "tone_cues": tone,
            "previous_sentiment": previous_sentiment,
            "sarcasm_score": sarcasm["score"],
            "sarcasm_level": sarcasm["level"],
            "sarcasm_signals": sarcasm["signals"],
            "possible_sarcasm": sarcasm["score"] >= self.SARCASM_THRESHOLD,
        }

    # ---- slang -------------------------------------------------------------
    def _find_slang(self, text: str):
        found: List[Dict[str, Any]] = []
        normalized = text
        # "r" is only slang for "are" when followed by u/you/ya ("r u ok").
        r_pat = re.compile(_L + r"r(?=\s+(?:u|you|ya)\b)", re.IGNORECASE)
        if r_pat.search(normalized):
            found.append({"term": "r", "meaning": "are", "polarity": 0.0})
            normalized = r_pat.sub("are", normalized)
        for slang, meaning in self.SLANG_MAP.items():
            pattern = _term_re(slang)
            if pattern.search(normalized):
                found.append({"term": slang, "meaning": meaning, "polarity": 0.0})
                normalized = pattern.sub(meaning, normalized)
        for pattern, term, meaning, polarity in self.SLANG_TERMS:
            if pattern.search(text):
                # "cap" alone must not fire when it is part of "no cap".
                if term == "cap" and _term_re("no cap").search(text):
                    continue
                found.append({"term": term, "meaning": meaning, "polarity": polarity})
        return found, normalized

    # ---- emoji -------------------------------------------------------------
    def _find_emoji(self, text_no_modifiers: str) -> List[Dict[str, Any]]:
        table = {_MODIFIERS.sub("", k): v for k, v in self.EMOJI_CUES.items()}
        counts: Dict[str, int] = {}
        order: List[str] = []
        for ch in text_no_modifiers:
            if ch in table:
                if ch not in counts:
                    order.append(ch)
                counts[ch] = counts.get(ch, 0) + 1
        cues = []
        for ch in order:
            meaning, polarity = table[ch]
            cues.append(
                {"emoji": ch, "meaning": meaning, "count": counts[ch],
                 "polarity": polarity}
            )
        for emoticon, (meaning, polarity) in self.EMOTICONS.items():
            if emoticon in text_no_modifiers:
                cues.append({"emoji": emoticon, "meaning": meaning, "count": 1,
                             "polarity": polarity})
        return cues

    # ---- tone --------------------------------------------------------------
    @staticmethod
    def _tone_cues(text: str) -> List[str]:
        cues = []
        if re.match(r"^\s*(ok|okay|k|fine|sure|whatever|noted)\s*\.+\s*$", text, re.I):
            cues.append("curt acknowledgement; can be passive-aggressive")
        return cues

    # ---- sarcasm -----------------------------------------------------------
    def _positive_before_negative(self, tokens: List[str]) -> bool:
        """Positive word followed (same clause, <= 8 words) by an un-negated
        negative-situation word, e.g. 'great, the app crashed again'."""
        words = [(i, t) for i, t in enumerate(tokens) if re.match(r"[a-z0-9']+$", t)]
        for wi, (ti, tok) in enumerate(words):
            if tok not in self.POSITIVE_WORDS or self._negated(words, wi):
                continue
            seen = 0
            for wj in range(wi + 1, len(words)):
                tj, tokj = words[wj]
                # clause boundary or contrast word ends the search
                if any(t in ".!?;" for t in tokens[ti:tj]) or tokj in self.CONTRAST:
                    break
                seen += 1
                if seen > 8:
                    break
                if self.NEGATIVE_SITUATION.match(tokj) and not self._negated(words, wj):
                    return True
        return False

    def _negated(self, words, idx: int) -> bool:
        for k in range(max(0, idx - 2), idx):
            w = words[k][1].replace("'", "")
            if w in self.NEGATORS or w.endswith("nt"):
                return True
        return False

    def _score_sarcasm(self, original, cleaned, emoji_cues, previous_sentiment):
        lower = cleaned.lower()
        tokens = re.findall(r"[a-z0-9']+|[.!?;,]", lower)
        word_tokens = [t for t in tokens if re.match(r"[a-z0-9']+$", t)]
        has_positive = any(
            t in self.POSITIVE_WORDS and not self._negated(
                [(0, w) for w in word_tokens], i)
            for i, t in enumerate(word_tokens)
        )
        signals: List[Dict[str, Any]] = []

        def add(name: str, weight: float):
            signals.append({"signal": name, "weight": weight})

        # 1. explicit marker
        if re.search(r"(?<!\S)/s(?![\w/])", original):
            add("explicit /s marker", 0.9)
        # 2. stock sarcastic phrases (word-bounded)
        for phrase in self.SARCASM_PHRASES:
            if re.search(_L + re.escape(phrase) + r"(?![\w])", lower):
                add(f"phrase: '{phrase}'", 0.7)
                break
        # 3. positive wording about a negative situation
        if self._positive_before_negative(tokens):
            add("positive wording + negative situation", 0.5)
        # 4. quoted 'positive' words:  thanks for the "help"
        if re.search(r"[\"“”](?:help\w*|great|nice|fix\w*|support|service|smart|genius|perfect)[\"“”]", lower):
            add("scare-quoted positive word", 0.5)
        # 5. sarcasm-leaning emoji
        emoji_weights = {"🙄": 0.6, "🙃": 0.45, "😒": 0.3, "-_-": 0.3}
        for cue in emoji_cues:
            w = emoji_weights.get(cue["emoji"])
            if w:
                add(f"emoji {cue['emoji']}", w)
                if has_positive:
                    add("positive wording clashes with emoji", 0.3)
        # 6. weak context: bare positive reply right after a negative message
        if (previous_sentiment == "Negative" and has_positive
                and 0 < len(word_tokens) <= 4):
            add("short positive reply after negative message", 0.3)

        # Boosters only strengthen an existing signal, never create one.
        if signals:
            if any(w.isupper() and len(w) >= 3 and w.lower() in self.POSITIVE_WORDS
                   for w in re.findall(r"[A-Za-z']+", original)):
                add("ALL-CAPS positive word", 0.15)
            if re.search(r"!{2,}", original):
                add("repeated exclamation marks", 0.1)
            if re.search(r"([a-z])\1{2,}", lower):
                add("elongated word", 0.1)

        remaining = 1.0
        for s in signals:
            remaining *= 1.0 - s["weight"]
        score = round(min(1.0 - remaining, 0.95), 3)
        level = ("likely" if score >= self.SARCASM_LIKELY
                 else "possible" if score >= self.SARCASM_THRESHOLD else "none")
        return {"score": score, "level": level, "signals": signals}
