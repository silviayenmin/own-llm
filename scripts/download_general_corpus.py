import os
import urllib.request
from pathlib import Path

def download_wikitext():
    # Setup directory
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "huge_corpus_general.txt"

    print("Preparing to download WikiText-2 general corpus...")

    # Public URLs for WikiText-2 parts (Train, Valid, Test)
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    ]

    combined_text = []

    for url in urls:
        filename = url.split("/")[-1]
        print(f"Downloading {filename} from PyTorch repository...")
        try:
            # Open URL and read text
            with urllib.request.urlopen(url) as response:
                content = response.read().decode("utf-8")
                # Clean up formatting (Wikitext files have double spaces or missing carriage returns)
                cleaned_content = content.replace(" \n", "\n")
                combined_text.append(cleaned_content)
                print(f"Successfully loaded {filename} ({len(cleaned_content)} characters)")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return

    # Join and write to target path
    print("Combining and writing to data/raw/huge_corpus_general.txt...")
    try:
        final_text = "\n\n".join(combined_text)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(final_text)
        
        file_size_mb = os.path.getsize(target_file) / (1024 * 1024)
        print(f"Success! General corpus saved to {target_file} ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"Error saving combined corpus: {e}")

if __name__ == "__main__":
    download_wikitext()
