import random
import re
from typing import Any, Dict, Optional


class Chatbot:
    """Simple rule-based chatbot with lightweight response refinement.

    Precedence (first match wins):
      1. Sarcasm / irony  (so "thanks for nothing" is never answered with
         "you're welcome")
      2. Bare greeting    (a greeting followed by a real message is NOT
         treated as a plain greeting: "hello, my order is late")
      3. Thanks           (only genuine thanks; "thanks but..." gets an
         apology instead)
      4. Intent phrases (doubt, explain, confused, help)
      5. Curt reply after a negative message (check-in)
      6. Emotion cues from emoji / slang
      7. Plain sentiment response
    """

    GREETING_ONLY = re.compile(
        r"^\s*(hello|hi|hey|greetings|hiya|yo)(\s+(there|bot|team|everyone))?\s*[!.?]*\s*$"
    )
    GREETING_PREFIX = re.compile(r"^\s*(hello|hi|hey|greetings|hiya|yo)\b[\s,!.-]*")

    def __init__(self):
        self.greeting_responses = [
            "Hello! How can I help you today?",
            "Hi there! What's on your mind?",
            "Hey! What can I help you with?"
        ]
        self.positive_responses = [
            "I'm glad to hear that!",
            "That's wonderful!",
            "Great to hear that!",
            "Excellent! How else can I help?"
        ]
        self.negative_responses = [
            "I'm sorry things aren't going well. Want to tell me a little more?",
            "That sounds frustrating. How can I help?",
            "I hear you. Let's see if we can make this easier."
        ]
        self.neutral_responses = [
            "I understand. Can you tell me more?",
            "I see. What would you like to know?",
            "Got it. What can I help you with?"
        ]
        self.likely_sarcasm_responses = [
            "That sounds like it was said with a bit of sarcasm, and it seems "
            "something has gone wrong. What happened?",
            "I'm reading that as frustration rather than praise. Sorry about "
            "that. What went wrong?",
        ]
        self.possible_sarcasm_responses = [
            "I might be misreading you, but that sounds like it could be "
            "ironic. Did something frustrating happen?",
            "Just checking: is that genuine, or is something not working "
            "the way it should?",
        ]
        self.anger_responses = [
            "I can tell you're really frustrated, and I'm sorry. Let's sort "
            "this out. What's the main problem right now?",
        ]
        self.amused_responses = [
            "Haha, glad that landed! Anything else I can help with?",
            "Love the energy! What else can I do for you?",
        ]
        self.warm_responses = [
            "That's so kind of you. I'm glad I could help!",
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _emoji_meanings(interp: Dict[str, Any]) -> str:
        return " ".join(c.get("meaning", "") for c in interp.get("emoji_cues", []))

    @staticmethod
    def _slang_terms(interp: Dict[str, Any]):
        return {s.get("term") for s in interp.get("slang", [])}

    def get_response(
        self,
        user_message: str,
        sentiment: str,
        interpretation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response using sentiment and optional language cues."""
        interp = interpretation or {}
        user_lower = user_message.lower()

        # 1. Sarcasm / irony first.
        if interp.get("possible_sarcasm"):
            pool = (self.likely_sarcasm_responses
                    if interp.get("sarcasm_level") == "likely"
                    else self.possible_sarcasm_responses)
            return random.choice(pool)

        # 2. Greeting: only a bare greeting gets a greeting reply. Otherwise
        #    drop the greeting word and respond to what the person said.
        if self.GREETING_ONLY.match(user_lower):
            return random.choice(self.greeting_responses)
        user_lower = self.GREETING_PREFIX.sub("", user_lower, count=1) or user_lower

        # 3. Thanks.
        if any(word in user_lower for word in ["thank", "thanks", "appreciate"]):
            mixed = re.search(r"\b(but|however|though|nothing)\b", user_lower)
            if mixed or sentiment == "Negative":
                return ("Sorry that didn't fully help. "
                        "What's still not working for you?")
            if "warm" in self._emoji_meanings(interp) or "affection" in self._emoji_meanings(interp):
                return random.choice(self.warm_responses)
            return "You're welcome! Anything else I can help with?"

        # 4. Intent recognition
        if any(phrase in user_lower for phrase in [
            "i have a doubt", "i have a question", "i want to ask"
        ]):
            return "Of course! What's your doubt? I'll be happy to help."

        if any(phrase in user_lower for phrase in [
            "can you explain", "explain this", "help me understand"
        ]):
            return "Absolutely! What would you like me to explain?"

        if any(phrase in user_lower for phrase in [
            "i'm confused", "i am confused", "i don't understand",
            "i do not understand"
        ]):
            return "No worries! Let's break it down step by step."

        if any(phrase in user_lower for phrase in ["help me", "i need help"]):
            return "Sure! Tell me what you need help with."

        # 5. Curt reply ("ok.", "fine.") after a negative message.
        if interp.get("tone_cues") and interp.get("previous_sentiment") == "Negative":
            return ("Just checking in: does that work for you, "
                    "or is something still off?")

        # 6. Emotion cues from emoji / slang.
        emoji_meanings = self._emoji_meanings(interp)
        slang = self._slang_terms(interp)
        if "anger" in emoji_meanings or "wtf" in slang:
            if sentiment == "Negative":
                return random.choice(self.anger_responses)
        if "annoyance" in emoji_meanings and sentiment != "Negative":
            return ("I noticed the tone might be a little frustrated. "
                    "Want to tell me what's going on?")
        if "amusement" in emoji_meanings and sentiment != "Negative":
            return random.choice(self.amused_responses)
        if sentiment == "Positive" and slang & {"slay", "goat", "fire", "slaps",
                                                 "banger", "goated", "lit", "sick"}:
            return random.choice(self.amused_responses)
        if sentiment == "Negative" and slang & {"mid", "cringe", "fml", "ugh", "yikes"}:
            return "Sounds like that was disappointing. What would have made it better?"

        # 7. Plain sentiment.
        if sentiment == "Positive":
            return random.choice(self.positive_responses)
        if sentiment == "Negative":
            return random.choice(self.negative_responses)
        return random.choice(self.neutral_responses)
