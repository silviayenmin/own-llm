import json
import random
from pathlib import Path

def generate_general_dataset():
    qa_pairs = []

    # 1. Diverse Greetings & Social Chat (250 pairs)
    greetings = [
        "Hello, how can I help you?", "Hi! What can I assist you with today?",
        "Hey! How is everything going?", "Good day! How may I help you?",
        "Hello! I am a general conversational assistant. What's on your mind?",
        "Hi there! Ready to chat. What would you like to discuss?"
    ]
    social_inputs = [
        "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
        "How are you?", "How is it going?", "What are you doing?", "Are you online?",
        "Tell me about yourself.", "Who are you?", "What is your purpose?"
    ]
    for _ in range(250):
        inp = random.choice(social_inputs)
        if "who" in inp.lower() or "yourself" in inp.lower():
            response = "I am a custom GPT assistant trained to help you with general chat, writing, and coding."
        elif "purpose" in inp.lower():
            response = "My purpose is to assist you with tasks, answer questions, and have helpful conversations."
        else:
            response = random.choice(greetings)
        qa_pairs.append({"prompt": inp, "response": response})

    # 2. General Knowledge & Facts (250 pairs)
    facts = [
        ("What is the capital of USA?", "The capital of the USA is Washington, D.C."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
        ("What is photosynthesis?", "Photosynthesis is the process used by plants to convert light energy into chemical energy."),
        ("How many continents are there?", "There are seven continents on Earth."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
        ("Which ocean is the largest?", "The Pacific Ocean is the largest ocean on Earth.")
    ]
    for _ in range(250):
        qa = random.choice(facts)
        qa_pairs.append({"prompt": qa[0], "response": qa[1]})

    # 3. Simple Reasoning & Logical Scenarios (200 pairs)
    riddles = [
        ("If you have three apples and take away two, how many do you have?", "You have two apples (the ones you took away)."),
        ("What has keys but can't open locks?", "A piano (or a keyboard) has keys but cannot open locks."),
        ("What gets wetter the more it dries?", "A towel gets wetter the more it dries."),
        ("If a red house is made of red bricks, what is a green house made of?", "A green house is made of glass."),
        ("What has a neck but no head?", "A bottle has a neck but no head."),
        ("What month has 28 days?", "All months have at least 28 days.")
    ]
    for _ in range(200):
        qa = random.choice(riddles)
        qa_pairs.append({"prompt": qa[0], "response": qa[1]})

    # 4. Text Manipulation & Writing Tasks (200 pairs)
    writing_tasks = [
        ("Write an email subject line for a meeting invitation.", "Meeting Invitation: Project Review and Discussion"),
        ("Summarize: The dog went to the park, chased a ball, played with friends, and fell asleep.", "A dog had a fun day playing at the park and then went to sleep."),
        ("Rephrase: It is raining cats and dogs.", "It is raining very heavily outside."),
        ("Write a polite greeting for a client.", "Dear Valued Client, I hope this message finds you well. How can we assist you today?"),
        ("How do I write a formal letter opening?", "You can open a formal letter with 'Dear Mr./Ms. [Last Name],' or 'To Whom It May Concern:'."),
        ("Translate to simple French: Hello, my friend.", "In French, you say: Bonjour, mon ami.")
    ]
    for _ in range(200):
        qa = random.choice(writing_tasks)
        qa_pairs.append({"prompt": qa[0], "response": qa[1]})

    # 5. Programming & Debugging Assistance (150 pairs)
    programming = [
        ("How do I print a string in Python?", "You print a string using the print() function, like this: print(\"Hello World\")"),
        ("What is the difference between a list and a tuple?", "Lists are mutable (can be changed), while tuples are immutable (cannot be changed)."),
        ("How do I write an if-else statement in Python?", "Example:\nif x > 0:\n    print('Positive')\nelse:\n    print('Non-positive')"),
        ("What is a syntax error?", "A syntax error occurs when the code violates the grammatical rules of the programming language."),
        ("How do I read a file in Python?", "You can read a file using a with-block: with open('file.txt', 'r') as f: data = f.read()"),
        ("What is Git?", "Git is a distributed version control system designed to track changes in source code.")
    ]
    for _ in range(150):
        qa = random.choice(programming)
        qa_pairs.append({"prompt": qa[0], "response": qa[1]})

    # Shuffle and slice to 1050
    random.shuffle(qa_pairs)
    qa_pairs = qa_pairs[:1050]

    # Write as data/instruct/qa_dataset.json (overwriting existing for finetune script)
    output_path = Path("data/instruct/qa_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(qa_pairs)} general conversational Q&A pairs and saved to {output_path}")

if __name__ == "__main__":
    generate_general_dataset()
