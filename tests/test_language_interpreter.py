from src.language_interpreter import LanguageInterpreter


def test_slang_normalization():
    result = LanguageInterpreter().interpret("tbh idk what to do rn")
    assert "to be honest" in result["normalized_text"]
    assert "I do not know" in result["normalized_text"]
    assert len(result["slang"]) >= 3


def test_emoji_interpretation():
    result = LanguageInterpreter().interpret("That is great 🙄")
    assert any(item["emoji"] == "🙄" for item in result["emoji_cues"])
    assert result["possible_sarcasm"] is True


def test_possible_sarcasm_phrase():
    result = LanguageInterpreter().interpret("Just what I needed today")
    assert result["possible_sarcasm"] is True


def test_plain_message_no_sarcasm_flag():
    result = LanguageInterpreter().interpret("Can you explain this topic?")
    assert result["possible_sarcasm"] is False
