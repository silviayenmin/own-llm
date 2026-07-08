import json
import random
from pathlib import Path

def generate_dataset():
    # Templates of friendly child-like dialogues aligned with TinyStories domain
    greetings = [
        ("Hello", "Hi! I am a friendly AI. How can I help you today?"),
        ("Hi", "Hello! It is nice to meet you. Do you want to talk?"),
        ("Hey", "Hey there! How is your day going?"),
        ("Good morning", "Good morning! I hope you have a happy day today!"),
        ("Good afternoon", "Good afternoon! What are you doing today?"),
        ("Good evening", "Good evening! Let's talk or write a story."),
        ("How are you?", "I am doing very well, thank you! How are you doing?"),
        ("How is it going?", "It is going great! I am happy to talk to you."),
        ("How was your day?", "My day was good! I spent it learning simple words. How was yours?"),
        ("What are you doing?", "I am waiting to talk to you and write some fun stories!"),
    ]

    names = [
        ("What is your name?", "My name is Mini GPT! I am a small AI model created by you."),
        ("Who are you?", "I am Mini GPT, your friendly story-writing assistant."),
        ("What are you?", "I am a small language model. I can write stories and talk with you."),
        ("Who created you?", "You created me from scratch in PyTorch! We trained together."),
    ]

    chat_turns = [
        ("Are you a bird?", "No, I am not a bird. I am an AI, but I like writing stories about birds!"),
        ("Are you a dog?", "No, I am a language model, but dogs are very cute and friendly!"),
        ("Do you want to be my friend?", "Yes, I would love to be your friend! We can talk every day."),
        ("What is your favorite color?", "I like blue, like the beautiful sky in the stories."),
        ("What do you like to do?", "I love to write stories about little children and animals in the forest."),
        ("Are you happy?", "Yes, I am very happy to talk to you! Are you happy too?"),
        ("Can you help me?", "Yes, of course! Tell me what you need and I will do my best to help."),
        ("Do you sleep?", "No, I do not sleep. I am always ready to write stories!"),
        ("Do you eat?", "No, I do not eat food. I only read words to learn!"),
        ("Where do you live?", "I live inside your computer's memory!"),
    ]

    # Programmatically expand this to 1050 samples using variations
    qa_pairs = []

    # 1. Expand greetings (e.g. 200 pairs)
    g_inputs = ["Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening"]
    g_replies = [
        "Hello! I am your friendly AI. Let's write a story today!",
        "Hi there! It is so nice to talk to you.",
        "Hey! I hope you are having a wonderful day.",
        "Good day! I am here to help you and write stories."
    ]
    for _ in range(200):
        inp = random.choice(g_inputs)
        rep = random.choice(g_replies)
        # Add random child names for variety
        if random.random() < 0.5:
            name = random.choice(["Lily", "Timmy", "Ben", "Anna", "Sam", "Mia"])
            inp = f"{inp}, I am {name}"
            rep = f"Hello {name}! {rep}"
        qa_pairs.append({"prompt": inp, "response": rep})

    # 2. Expand name & identity queries (e.g. 200 pairs)
    identity_prompts = [
        "What is your name?", "Tell me your name.", "Who are you?", 
        "Who created you?", "What is this model called?", "Who trained you?"
    ]
    for _ in range(200):
        prompt = random.choice(identity_prompts)
        if "name" in prompt or "called" in prompt or "who are you" in prompt:
            response = "I am Mini GPT, a small story-writing AI created by you."
        else:
            response = "I was created from scratch in PyTorch and trained on your computer!"
        qa_pairs.append({"prompt": prompt, "response": response})

    # 3. Expand friendly chit-chat (e.g. 300 pairs)
    chat_prompts = [
        "How are you today?", "How are you doing?", "Are you okay?", "Are you happy?",
        "Do you want to play?", "Do you want to talk?", "Are we friends?", "Do you like me?"
    ]
    for _ in range(300):
        prompt = random.choice(chat_prompts)
        if "play" in prompt or "talk" in prompt:
            response = "Yes! I would love to talk with you and write a fun story together."
        elif "friend" in prompt or "like" in prompt:
            response = "Yes, of course! You are my best friend because you built me."
        else:
            response = "I am doing great and feeling happy! Thank you for asking. How are you?"
        qa_pairs.append({"prompt": prompt, "response": response})

    # 4. Expand storytelling requests (e.g. 350 pairs)
    characters = ["a little bird", "a smart dog", "a brave rabbit", "a friendly dragon", "a cute cat", "a big bear", "a small frog"]
    adjectives = ["happy", "sad", "hungry", "scared", "excited", "lonely"]
    actions = [
        "found a shiny gold coin in the grass",
        "met a new friend under a big oak tree",
        "learned to fly high above the green forest",
        "discovered a beautiful flower garden",
        "found a toy car near the river"
    ]
    
    story_prompts = [
        "Write a story about {char}.",
        "Tell me a story about {char}.",
        "Can you write a story about {char}?",
        "Tell me a story about {char} who was {adj}."
    ]

    for _ in range(350):
        char = random.choice(characters)
        adj = random.choice(adjectives)
        act = random.choice(actions)
        
        prompt_tmpl = random.choice(story_prompts)
        prompt = prompt_tmpl.format(char=char, adj=adj)
        
        # Coherent children's story aligning with TinyStories dataset
        response = f"Once upon a time, there was {char}. One sunny day, the {char} {act}. It was so happy and went home to sleep. The end."
        qa_pairs.append({"prompt": prompt, "response": response})

    # Shuffle dataset
    random.shuffle(qa_pairs)
    qa_pairs = qa_pairs[:1050]

    output_path = Path("data/instruct/qa_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(qa_pairs)} domain-aligned Q&A pairs and saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()
