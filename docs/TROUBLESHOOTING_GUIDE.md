# 🛠️ own-llm: Troubleshooting & Learning Guide (Tanglish version)

Indha file-la namma train pannumbodhu face panna issues, adhai namma epdi solve pannom, and inime thirumba indha issues vandha epdi handle pannanum-nu clear-aah eludhi iruken.

---

## 1. 🔍 Why We Struggled? (Enna Prachana Aachu?)

Namma training flow-la 2 main issues-naala struggle pannom:

### Issue 1: Model Output-la Garbage Characters Vandhadhu
*   **Enna aachu:** Base model-a pretrain panni mudichuttu text generate panni paarkumbodhu, full-aah garbage text (like `^5=;9F@=...`) prediction-aah vandhadhu.
*   **Reason:** Model code-la vocab size-a `4096` BPE parameters vechu train pannom. Aana data preprocess panni binary files (`train.bin` and `val.bin`) generate pannumbodhu, model config-la character-level tokenizer parameters apply aayirundhadhu. 
*   **Logical Mismatch:** Character-level binary-a BPE tokenizer parameters load panni decode panna try pannadhunaala, token IDs collide aagi full-aah garbage output generate aachu.

### Issue 2: Preprocessing Run Aagum Podhu Silent Crash Aachu
*   **Enna aachu:** Dataset pre-processing run pannumpodhu `0.29 MB` or `1.46 MB` state-la terminal logs stop aagi, endha oru error message-um illaama terminal automatic-a close aayiruchu.
*   **Reason:** Tokenizer code-la `_encode_chunk` method recursive-aah run aayirundhadhu. Windows-la Python 3.12 memory limit-naala high recursion call stacks oodumbodhu crash aachu. Plus, pure Python loops-la billions of short-lived list/tuple allocations recreate aaguradhunaala, Python garbage collector run aagum podhu heap-a corrupt panni `python312.dll` access violation error (`0xc0000005`) trigger panni python-a crash panniduchu.

---

## 2. 🏆 How We Solved It? (Epdi Resolve Pannom?)

Namma panna permanent code updates (Romba easy-aah puriyra mari):

### 1. Iterative Tokenizer with Word-level Caching (Sticky Note Analogy):
*   **Pazhaya Method (Slow & Crash):** 
    Oru 30MB book-la "Once" nu oru word 10,000 times varudhuna, namma tokenizer andha word-a paarkura ovvoru thadavayum BPE merging rules-a mudhalla irundhu calculate pannum. Idhu epdi na, oru word-oda meaning-a therinjuka, book-la andha word varumpodhulaam dictionary-a thirumba thirumba search panra mari. Idhala system load aagi crash aayiruchu.
*   **Pudhusa namma panna fix:**
    First time oru word (e.g., "Once") varumpodhu, adha calculate panni values-a BPE list-a convert panni oru temporary `self.cache` (sticky note) la eludhi vachupom. Next time andha word "Once" thirumba vandha, dictionary-a thoda matom; direct-a sticky note parameters pathu instant-a value-a eduthupom. 
    Indhunaala training pre-processing timing **50 minutes-la irundhu just 10 seconds-aah** koranjiruchu!

### 2. Disabling Garbage Collector (Sweeper Analogy):
*   **Garbage Collector-na enna?**
    Python software run aagumpodhu create aagura waste variables memory-a clean panna system backend-la oru cleaner (Garbage Collector) automatic-a sweep pannite iruparu.
*   **Pazhaya Method (Crash aana reason):**
    Namma tokenizer raw text-a encode pannumpodhu seconds-la millions of tiny memory variables generate aachudhula. Namma fast-a variables create panni velai senjitu irukumpodhu, background cleaner-um fast-a clean pannanum-nu try panni system file (`python312.dll` error) kulla mothi python block collapse aayiruchu.
*   **Pudhusa namma panna fix:**
    Tokenizer run aagum podhu backend cleaner-a pause panta mari `gc.disable()` panni run pannom. "Naan indha 10 seconds preprocessing loop mudikura varai nee room-a clean panna koodadhu" nu disable pannitom. Loop mudinjadhum clean panna instructions trigger panna `gc.enable()` check trigger pannitom. So, crash clear aagi system stability permanent solve aayiruchu!

---

## 3. 🛡️ If It Happens Again, How to Deal with it? (Thirumba Vandha Enna Pannanum?)

Future-la thirumbavum python command run aagum podhu silent crash trigger aachu-na:

### Step 1: Check Windows Event Logs for Python Error
Terminal-la indha command-a run panni error parameters check panni paaru:
```powershell
Get-EventLog -LogName Application -EntryType Error -Newest 5 | Format-List
```
*   Crash target application `python.exe` error code `0xc0000005` in `python312.dll` check off aana memory allocation GC mismatch aayirukunu artham.

### Step 2: Implement GC Controls and Caching
*   Namma dynamic-a dynamic object structures high allocation loops execute pannum podhu, pre-processing-kulla temporary-a garbage collection-a disable panni run pannanum:
    ```python
    import gc
    gc.disable()
    # High object allocation loops run here (e.g. tokenizer encoding loops)
    gc.enable()
    ```
*   And high processing methods recursion skip panni sequential loop structures iterative method logic scale panni caching maps build pannanum.

---

## 4. 🧠 BPE Tokenizer Training Memory Limit Crash (Tanglish version)

*   **Enna aachu:** BPE tokenizer training script `tokenizer.train_tokenizer` execute pannumpodhu prompt silent-a thirumba vandhudum. Aana `tokenizer.json` directory general-kulla save aagi irukadhu, and output prints complete aagadhu.
*   **Reason (Enna prachana):** Pure Python loops-la 16,127 merge iterations-la millions of temporary tuples/dicts dynamic-a create aaguradhunaala, memory heap crash trigger aagi Windows system Python process-a warning illaama silent-a kill pannidum.
*   **Solution (Epdi thirumba vandha fix panradhu):**
    1.  **Reduce BPE Text Sample Size:** Tokenizer input-a raw data-la full-aah 13MB load pannama first **100KB or 200KB**-a crop panni limit panni run pannanum:
        ```python
        # Target limit configuration check (100KB size)
        text = text[:100 * 1024]
        ```
    2.  **Enable/Disable GC around training loop:** Training speed increment use panna memory GC loops disable control setup use pannanum.
    3.  **Command to run:**
        ```powershell
        python -m tokenizer.train_tokenizer --type bpe --vocab_size 16384 --raw_data_path data/raw/huge_corpus_general.txt --tokenizer_dir data/tokenizer_general
        ```