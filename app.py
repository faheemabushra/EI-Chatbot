import streamlit as st
from src.sentiment_analyzer import SentimentAnalyzer
from src.language_interpreter import LanguageInterpreter
from src.conversation_manager import ConversationManager
from src.chatbot import Chatbot

st.set_page_config(page_title="EI Chatbot", page_icon="💬", layout="centered")

if "analyzer" not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if "interpreter" not in st.session_state:
    st.session_state.interpreter = LanguageInterpreter()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 EI Chatbot")
st.caption("A chatbot prototype that refines responses using sentiment and informal-language cues.")

with st.sidebar:
    st.header("Conversation")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation = ConversationManager()
        st.rerun()

    conversation = st.session_state.conversation
    if conversation.get_user_messages():
        overall = conversation.get_overall_sentiment()
        st.metric("Overall sentiment", overall["label"], f"{overall['compound']:+.2f}")
        st.write(f"Trend: **{conversation.get_sentiment_trend()}**")
        st.write(
            f"👍 {overall['positive_count']}  ·  😐 {overall['neutral_count']}  ·  "
            f"👎 {overall['negative_count']}"
        )
    st.caption("Interpretation is heuristic and may not correctly identify every sarcastic message.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("analysis"):
            st.caption(message["analysis"])

if prompt := st.chat_input("Send a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    conversation = st.session_state.conversation
    interpretation = st.session_state.interpreter.interpret(
        prompt, previous_sentiment=conversation.last_user_label()
    )
    # VADER scores the user's own wording with emoji removed (it misreads
    # emoji such as 💀 by their literal names). Slang polarity, emoji and
    # possible sarcasm then adjust the score. Slang is not expanded before
    # scoring, so "tbh" does not inject sentiment words.
    scores, label = st.session_state.analyzer.analyze_with_context(
        prompt, interpretation
    )
    conversation.add_message("user", prompt, scores, interpretation)
    response = st.session_state.chatbot.get_response(
        prompt, label, interpretation=interpretation
    )
    conversation.add_message("bot", response)

    details = [f"Sentiment: {label} ({scores['compound']:+.2f})"]
    if scores["adjustments"]:
        details.append(
            f"adjusted from raw {scores['raw_compound']:+.2f} "
            f"[{'; '.join(scores['adjustments'])}]"
        )
    if interpretation["slang"]:
        details.append("Slang: " + ", ".join(
            f"{item['term']} = {item['meaning']}"
            for item in interpretation["slang"]
        ))
    if interpretation["emoji_cues"]:
        details.append("Emoji cues: " + ", ".join(
            f"{item['emoji']}"
            + (f" x{item['count']}" if item["count"] > 1 else "")
            + f" ({item['meaning']})"
            for item in interpretation["emoji_cues"]
        ))
    if interpretation["tone_cues"]:
        details.append("Tone: " + "; ".join(interpretation["tone_cues"]))
    if interpretation["possible_sarcasm"]:
        details.append(
            f"Sarcasm/irony: {interpretation['sarcasm_level']} "
            f"(score {interpretation['sarcasm_score']:.2f}; "
            + ", ".join(s["signal"] for s in interpretation["sarcasm_signals"])
            + ") (heuristic)"
        )

    analysis_text = " | ".join(details)

    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(analysis_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "analysis": analysis_text
    })
    st.rerun()
