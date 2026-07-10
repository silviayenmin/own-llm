import sys
try:
    import sentence_transformers
    print("SUCCESS: sentence_transformers imported!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
