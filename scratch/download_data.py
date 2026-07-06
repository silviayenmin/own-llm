import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
output_path = "data/raw/input.txt"

print(f"Downloading Tiny Shakespeare dataset from:\n  {url}...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
urllib.request.urlretrieve(url, output_path)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"Successfully saved to {output_path} ({size_mb:.2f} MB).")
