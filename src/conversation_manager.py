from typing import List, Dict
from datetime import datetime
import json

class ConversationManager:
    """Manages conversation history and sentiment tracking"""
    
    def __init__(self):
        self.messages: List[Dict] = []
        
    def add_message(self, role: str, text: str, sentiment_scores: Dict = None,
                    interpretation: Dict = None):
        """Add a message to conversation history"""
        message = {
            'role': role,
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'sentiment': sentiment_scores,
            'interpretation': interpretation
        }
        self.messages.append(message)

    def last_user_label(self):
        """Sentiment label of the most recent user message (None if none).
        Used as weak context when judging sarcasm in the next message."""
        user_messages = self.get_user_messages()
        if not user_messages or not user_messages[-1].get('sentiment'):
            return None
        compound = user_messages[-1]['sentiment']['compound']
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"
        
    def get_user_messages(self) -> List[Dict]:
        """Get only user messages"""
        return [msg for msg in self.messages if msg['role'] == 'user']
    
    def get_overall_sentiment(self) -> Dict:
        """Calculate overall conversation sentiment"""
        user_messages = self.get_user_messages()
        
        if not user_messages:
            return {'compound': 0, 'label': 'Neutral', 'count': 0}
        
        total_compound = sum(msg['sentiment']['compound'] for msg in user_messages)
        avg_compound = total_compound / len(user_messages)
        
        # Count sentiment types
        positive = sum(1 for msg in user_messages if msg['sentiment']['compound'] >= 0.05)
        negative = sum(1 for msg in user_messages if msg['sentiment']['compound'] <= -0.05)
        neutral = len(user_messages) - positive - negative
        
        # Determine label
        if avg_compound >= 0.05:
            label = "Positive"
        elif avg_compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {
            'compound': avg_compound,
            'label': label,
            'count': len(user_messages),
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral
        }
    
    def get_sentiment_trend(self) -> str:
        """Analyze if sentiment is improving, declining, or stable"""
        user_messages = self.get_user_messages()
        
        if len(user_messages) < 2:
            return "Insufficient data"
        
        # Compare first half vs second half
        mid = len(user_messages) // 2
        first_half = user_messages[:mid]
        second_half = user_messages[mid:]
        
        avg_first = sum(msg['sentiment']['compound'] for msg in first_half) / len(first_half)
        avg_second = sum(msg['sentiment']['compound'] for msg in second_half) / len(second_half)
        
        diff = avg_second - avg_first
        
        if diff > 0.1:
            return "Improving"
        elif diff < -0.1:
            return "Declining"
        else:
            return "Stable"
    
    def export_to_json(self, filepath: str):
        """Export conversation to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2)
