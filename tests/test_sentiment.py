import pytest
from src.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()
    
    def test_positive_sentiment(self, analyzer):
        text = "I love this service! It's amazing!"
        scores, label = analyzer.analyze_with_label(text)
        assert label == "Positive"
        assert scores['compound'] > 0.05
    
    def test_negative_sentiment(self, analyzer):
        text = "This is terrible and disappointing"
        scores, label = analyzer.analyze_with_label(text)
        assert label == "Negative"
        assert scores['compound'] < -0.05
    
    def test_neutral_sentiment(self, analyzer):
        text = "The product is okay"
        scores, label = analyzer.analyze_with_label(text)
        assert label in ["Neutral", "Positive", "Negative"]
    
    def test_get_sentiment_label(self, analyzer):
        assert analyzer.get_sentiment_label(0.6) == "Positive"
        assert analyzer.get_sentiment_label(-0.6) == "Negative"
        assert analyzer.get_sentiment_label(0.02) == "Neutral"
