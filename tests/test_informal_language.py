"""Tests for sarcasm, slang, emoji handling and the contextual sentiment /
response refinements. Written without pytest-only features so they are easy
to read; run with `pytest`."""
from src.chatbot import Chatbot
from src.language_interpreter import LanguageInterpreter
from src.sentiment_analyzer import SentimentAnalyzer
from src.conversation_manager import ConversationManager

li = LanguageInterpreter()


def run(text, previous=None):
    """interpret -> contextual sentiment -> chatbot reply"""
    interp = li.interpret(text, previous_sentiment=previous)
    scores, label = SentimentAnalyzer().analyze_with_context(text, interp)
    reply = Chatbot().get_response(text, label, interpretation=interp)
    return interp, scores, label, reply


# ---- sarcasm: should be detected -----------------------------------------
def test_sarcasm_detected():
    for text in [
        "Oh great, another meeting",
        "Wow, thanks a lot for ruining my day",
        "I just love waiting 2 hours for support",
        "Great, the app crashed again",
        "Nice job breaking it",
        "Yeah, because that totally worked",
        "I love how you ignore my question",
        "Well that was helpful /s",
        "OH GREAT!!!",
        "That is great 🙄",
        "Thanks for the \"help\"",
    ]:
        assert li.interpret(text)["possible_sarcasm"], text


def test_sarcasm_levels_and_signals():
    r = li.interpret("Well that was helpful /s")
    assert r["sarcasm_level"] == "likely"
    assert any("/s" in s["signal"] for s in r["sarcasm_signals"])
    assert 0 < li.interpret("Great, the app crashed again")["sarcasm_score"] <= 0.95


# ---- sarcasm: must NOT be detected (false positives) ----------------------
def test_sarcasm_false_positives():
    for text in [
        "That plan was iffy",                       # substring of "as if"
        "I love this app, it never crashes",        # negated negative
        "I'm happy my order isn't late",
        "The delivery was late but the support was great",
        "Great service, thanks!",
        "Thanks, that really helped me a lot",
        "Can you explain this topic?",
        "I got an L on my exam",
    ]:
        assert not li.interpret(text)["possible_sarcasm"], text


def test_sarcasm_flips_sentiment_to_negative():
    for text in ["Oh great, another meeting", "Thanks for nothing",
                 "Great, the app crashed again"]:
        _, scores, label, _ = run(text)
        assert label == "Negative", (text, scores)
        assert scores["adjustments"]


# ---- slang ----------------------------------------------------------------
def test_single_letter_slang_does_not_corrupt_text():
    r = li.interpret("I got an L on my exam, r u ok")
    assert "an L on" in r["normalized_text"]
    assert "are you ok" in r["normalized_text"]
    assert "U.S. policy" in li.interpret("U.S. policy")["normalized_text"]


def test_modern_slang_detected_with_polarity():
    terms = {s["term"]: s["polarity"] for s in
             li.interpret("no cap this is the goat, lowkey fire")["slang"]}
    assert terms["goat"] > 0 and terms["fire"] > 0 and "no cap" in terms
    mid = {s["term"]: s["polarity"] for s in li.interpret("ur app is mid af")["slang"]}
    assert mid["mid"] < 0 and "af" in mid


def test_context_sensitive_slang_not_misread():
    for text in ["I feel sick", "the fire alarm went off", "I fed the goat"]:
        assert li.interpret(text)["slang"] == [], text
    assert any(s["term"] == "sick" for s in li.interpret("that's sick bro")["slang"])


def test_slang_shifts_sentiment():
    assert run("ur app is mid af")[2] == "Negative"
    assert run("no cap this is the goat")[2] == "Positive"


# ---- emoji ----------------------------------------------------------------
def test_emoji_variants_and_counts():
    assert li.interpret("❤ you")["emoji_cues"][0]["emoji"] == "❤"      # no VS16
    assert li.interpret("❤️ you")["emoji_cues"][0]["emoji"] == "❤"
    assert li.interpret("👍🏽 sure")["emoji_cues"][0]["emoji"] == "👍"   # skin tone
    assert li.interpret("😭😭😭")["emoji_cues"][0]["count"] == 3


def test_ambiguous_emoji_follow_the_text():
    assert run("I'm so happy 😭")[2] == "Positive"
    assert run("I'm so sad and upset 😭")[2] == "Negative"
    assert run("😭😭😭")[2] == "Neutral"      # no text to disambiguate


def test_emoji_only_messages():
    assert run("👍")[2] == "Positive"
    assert run("😡")[2] == "Negative"


# ---- tone -----------------------------------------------------------------
def test_curt_reply_cue():
    assert li.interpret("ok.")["tone_cues"]
    assert not li.interpret("ok")["tone_cues"]


# ---- chatbot precedence / refinement -------------------------------------
def test_sarcastic_thanks_is_not_answered_with_youre_welcome():
    for text in ["Thanks for nothing", "Wow, thanks a lot for ruining my day"]:
        assert "welcome" not in run(text)[3].lower(), text


def test_thanks_but_negative_gets_apology():
    reply = run("thanks but this is useless")[3]
    assert "welcome" not in reply.lower()


def test_genuine_thanks_still_welcomed():
    assert "welcome" in run("Thanks, that really helped me a lot")[3].lower()


def test_greeting_with_complaint_is_not_a_bare_greeting():
    bot = Chatbot()
    reply = run("Hello, my order is late")[3]
    assert reply not in bot.greeting_responses
    assert Chatbot().get_response("Hi!", "Neutral") in bot.greeting_responses


def test_curt_reply_after_negative_checks_in():
    reply = run("fine.", previous="Negative")[3]
    assert "checking in" in reply.lower()


def test_previous_sentiment_context_is_weak_only():
    # Context alone must never flag sarcasm
    assert not li.interpret("great", previous_sentiment="Negative")["possible_sarcasm"]


# ---- conversation manager -------------------------------------------------
def test_manager_tracks_adjusted_sentiment_and_last_label():
    manager = ConversationManager()
    assert manager.last_user_label() is None
    interp, scores, _, _ = run("Oh great, another meeting")
    manager.add_message("user", "Oh great, another meeting", scores, interp)
    assert manager.last_user_label() == "Negative"
    assert manager.get_overall_sentiment()["label"] == "Negative"
