from src.sentiment_analyzer import SentimentAnalyzer
from src.language_interpreter import LanguageInterpreter
from src.conversation_manager import ConversationManager
from src.chatbot import Chatbot
from src.utils import print_sentiment, print_header, print_summary, print_cues
from colorama import Fore, Style

def main():
    analyzer = SentimentAnalyzer()
    interpreter = LanguageInterpreter()
    conversation = ConversationManager()
    bot = Chatbot()
    
    print_header("CHATBOT WITH SENTIMENT ANALYSIS")
    print(f"{Fore.CYAN}Type 'quit', 'exit', or 'bye' to end the conversation.{Style.RESET_ALL}\n")
    
    while True:
        # Get user input
        user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
        
        if not user_input:
            continue
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print(f"\n{Fore.CYAN}Ending conversation...{Style.RESET_ALL}")
            break
        
        # Interpret slang / emoji / sarcasm, then score sentiment with those cues
        interpretation = interpreter.interpret(
            user_input, previous_sentiment=conversation.last_user_label()
        )
        scores, sentiment = analyzer.analyze_with_context(user_input, interpretation)
        conversation.add_message('user', user_input, scores, interpretation)

        # Display sentiment and detected cues
        print_sentiment(user_input, sentiment, scores['compound'])
        print_cues(interpretation, scores)

        # Generate bot response
        response = bot.get_response(user_input, sentiment, interpretation=interpretation)
        conversation.add_message('bot', response)

        print(f"{Fore.GREEN}Bot: {Style.RESET_ALL}{response}\n")
    
    # Tier 1: Overall sentiment analysis
    if conversation.get_user_messages():
        overall = conversation.get_overall_sentiment()
        trend = conversation.get_sentiment_trend()
        print_summary(overall, trend)
    else:
        print(f"{Fore.YELLOW}No messages to analyze.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
