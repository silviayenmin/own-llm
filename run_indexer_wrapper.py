import traceback
import sys

print("Starting indexing script wrapper...")
with open("rag_run.log", "w", encoding="utf-8") as f:
    f.write("Wrapper started\n")
    try:
        import rag_indexer
        f.write("Imported rag_indexer successfully\n")
        rag_indexer.build_index()
        f.write("build_index completed successfully!\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
        print("Wrapper failed!")
        sys.exit(1)
print("Wrapper finished successfully!")
