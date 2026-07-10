import os
import zipfile
import urllib.request
from pathlib import Path

def compile_corpus():
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "huge_corpus_general.txt"
    
    print("=== Step 1: Locating Existing TinyStories Corpus ===")
    tinystories_path = target_dir / "huge_corpus.txt"
    has_tinystories = tinystories_path.exists()
    if has_tinystories:
        tinystories_size_gb = os.path.getsize(tinystories_path) / (1024 * 1024 * 1024)
        print(f"Found TinyStories corpus: {tinystories_path} ({tinystories_size_gb:.2f} GB)")
    else:
        print("TinyStories corpus not found at data/raw/huge_corpus.txt. Skipping stories.")

    print("\n=== Step 2: Downloading WikiText-103 (500MB Wikipedia Corpus) ===")
    wikitext_zip = target_dir / "wikitext-103-raw-v1.zip"
    wikitext_extracted_dir = target_dir / "wikitext-103-raw"
    wikitext_file = wikitext_extracted_dir / "wiki.train.raw"
    
    if not wikitext_file.exists():
        url = "https://huggingface.co/datasets/segyges/wikitext-103/resolve/main/wikitext-103-raw-v1.zip"
        print(f"Downloading WikiText-103 from {url}...")
        try:
            # Download with progress indicator
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rDownloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")
            
            urllib.request.urlretrieve(url, wikitext_zip, progress)
            print("\nDownload finished! Extracting zip file...")
            
            with zipfile.ZipFile(wikitext_zip, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            print("Extraction completed successfully!")
            # Clean up zip file
            if wikitext_zip.exists():
                os.remove(wikitext_zip)
        except Exception as e:
            print(f"\nError downloading/extracting WikiText-103: {e}")
    else:
        print(f"WikiText-103 already exists at {wikitext_file}")

    print("\n=== Step 3: Scanning Local Python Virtual Environment (Code Syntax) ===")
    code_text = []
    venv_dir = Path("myenv")
    if venv_dir.exists():
        print(f"Scanning Python files in {venv_dir} to extract code grammar...")
        py_files = list(venv_dir.glob("**/*.py"))
        print(f"Found {len(py_files)} Python files. Reading syntax from first 2000 files...")
        
        count = 0
        for py_path in py_files[:2000]:
            try:
                # Read content safely
                with open(py_path, encoding="utf-8", errors="ignore") as f:
                    code_text.append(f.read())
                count += 1
            except Exception:
                continue
        print(f"Successfully loaded code from {count} python files.")
    else:
        print("Virtual environment 'myenv' not found. Skipping local code scan.")

    print("\n=== Step 4: Compiling Everything into huge_corpus_general.txt ===")
    print(f"Writing to {output_file}...")
    
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            # 1. Write Wikipedia (Facts)
            if wikitext_file.exists():
                print("Writing WikiText-103...")
                with open(wikitext_file, encoding="utf-8") as f:
                    out.write(f.read())
                out.write("\n\n")

            # 2. Write Python Code (Programming Logic)
            if code_text:
                print("Writing Python Code Syntax...")
                out.write("\n\n".join(code_text))
                out.write("\n\n")

            # 3. Write Storytelling (Narrative structure)
            if has_tinystories:
                print("Writing TinyStories...")
                # Stream it to prevent loading 1.8GB into memory at once
                with open(tinystories_path, encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
        
        final_size_gb = os.path.getsize(output_file) / (1024 * 1024 * 1024)
        print(f"\n🎉 Success! Combined General Corpus created at: {output_file}")
        print(f"Total File Size: {final_size_gb:.2f} GB")
        
    except Exception as e:
        print(f"Error compiling output file: {e}")

if __name__ == "__main__":
    compile_corpus()
