import json
import os

log_path = "results/gemma3_4b/household_audit.jsonl"
if not os.path.exists(log_path):
    print(f"File not found: {log_path}")
    exit(1)

from collections import deque

with open(log_path, 'r', encoding='utf-8') as f:
    # Read last line efficiently
    last_line = deque(f, 1)[0]
    data = json.loads(last_line)
    
    print(f"Checking Run ID: {data.get('run_id')}")
    print("JSON KEYS:", list(data.keys()))
    
    prompt = data.get("input", "MISSING_KEY")
    print(f"INPUT VALUE (Result type: {type(prompt)}): '{prompt}'")
    
    if prompt == "":
        print("⚠️ INPUT IS EMPTY STRING!")
