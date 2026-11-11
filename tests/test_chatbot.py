import pytest
from src.chatbot import Chatbot
from src.conversation_manager import ConversationManager

class TestChatbot:
    
    @pytest.fixture
    def bot(self):
        return Chatbot()
    
    def test_greeting_response(self, bot):
        response = bot.get_response("Hello", "Neutral")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_positive_response(self, bot):
        response = bot.get_response("Great service!", "Positive")
        assert isinstance(response, str)
    
    def test_negative_response(self, bot):
        response = bot.get_response("This is bad", "Negative")
        assert isinstance(response, str)

class TestConversationManager:
    
    @pytest.fixture
    def manager(self):
        return ConversationManager()
    
    def test_add_message(self, manager):
        manager.add_message('user', 'Hello', {'compound': 0.5})
        assert len(manager.messages) == 1
    
    def test_get_user_messages(self, manager):
        manager.add_message('user', 'Hi', {'compound': 0.3})
        manager.add_message('bot', 'Hello!', None)
        user_msgs = manager.get_user_messages()
        assert len(user_msgs) == 1
        assert user_msgs[0]['role'] == 'user'
    
    def test_overall_sentiment(self, manager):
        manager.add_message('user', 'Great!', {'compound': 0.6})
        manager.add_message('user', 'Terrible!', {'compound': -0.6})
        overall = manager.get_overall_sentiment()
        assert overall['count'] == 2
        assert overall['label'] == "Neutral"


def test_this_does_not_trigger_greeting():
    from src.chatbot import Chatbot
    bot = Chatbot()
    response = bot.get_response("I absolutely love this!", "Positive")
    assert response != "Hi there! What's on your mind?"
    assert response in bot.positive_responses


def test_actual_greeting_is_recognized():
    from src.chatbot import Chatbot
    bot = Chatbot()
    response = bot.get_response("Hi!", "Neutral")
    assert response in bot.greeting_responses
