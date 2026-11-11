from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def print_sentiment(text: str, sentiment: str, score: float):
    """Print message with color-coded sentiment"""
    color = Fore.WHITE
    
    if sentiment == "Positive":
        color = Fore.GREEN
    elif sentiment == "Negative":
        color = Fore.RED
    elif sentiment == "Neutral":
        color = Fore.YELLOW
    
    print(f"{color}→ Sentiment: {sentiment} (Score: {score:.2f}){Style.RESET_ALL}")

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Fore.CYAN}{'=' * 50}")
    print(f"{text:^50}")
    print(f"{'=' * 50}{Style.RESET_ALL}\n")

def print_summary(overall: dict, trend: str):
    """Print final conversation summary"""
    print_header("CONVERSATION SUMMARY")
    
    label_color = Fore.WHITE
    if overall['label'] == "Positive":
        label_color = Fore.GREEN
    elif overall['label'] == "Negative":
        label_color = Fore.RED
    elif overall['label'] == "Neutral":
        label_color = Fore.YELLOW
    
    print(f"{Fore.CYAN}Overall Sentiment:{Style.RESET_ALL} {label_color}{overall['label']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Average Score:{Style.RESET_ALL} {overall['compound']:.2f}")
    print(f"{Fore.CYAN}Sentiment Trend:{Style.RESET_ALL} {trend}")
    print(f"\n{Fore.CYAN}Message Breakdown:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Positive:{Style.RESET_ALL} {overall['positive_count']}")
    print(f"  {Fore.RED}Negative:{Style.RESET_ALL} {overall['negative_count']}")
    print(f"  {Fore.YELLOW}Neutral:{Style.RESET_ALL} {overall['neutral_count']}")
    print(f"  {Fore.CYAN}Total:{Style.RESET_ALL} {overall['count']}")


def print_cues(interpretation: dict, scores: dict = None):
    """Print slang / emoji / sarcasm cues detected in a message"""
    lines = []
    for item in interpretation.get("slang", []):
        lines.append(f"slang: {item['term']} = {item['meaning']}")
    for item in interpretation.get("emoji_cues", []):
        times = f" x{item['count']}" if item.get("count", 1) > 1 else ""
        lines.append(f"emoji: {item['emoji']}{times} = {item['meaning']}")
    for tone in interpretation.get("tone_cues", []):
        lines.append(f"tone: {tone}")
    if interpretation.get("possible_sarcasm"):
        lines.append(
            f"sarcasm: {interpretation['sarcasm_level']} "
            f"(score {interpretation['sarcasm_score']:.2f}; "
            + ", ".join(s["signal"] for s in interpretation["sarcasm_signals"]) + ")"
        )
    if scores and scores.get("adjustments"):
        lines.append(
            f"score: raw {scores['raw_compound']:+.2f} -> adjusted "
            f"{scores['compound']:+.2f} ({'; '.join(scores['adjustments'])})"
        )
    for line in lines:
        print(f"{Fore.MAGENTA}  · {line}{Style.RESET_ALL}")
