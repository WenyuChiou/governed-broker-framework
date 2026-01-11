import os
import subprocess
import shutil
import time

MODELS = ["llama3.2:3b", "gemma3:4b", "deepseek-r1:8b", "gpt-oss:latest"]
PROFILES = ["strict", "relaxed"]

def run_sim(model, profile):
    cmd = [
        "python", "examples/single_agent/run_experiment.py",
        "--model", model,
        "--num-agents", "100",
        "--num-years", "1",
        "--governance-profile", profile
    ]
    print(f"\n>>> STARTING: Model={model}, Profile={profile}")
    print(f">>> Command: {' '.join(cmd)}")
    
    start = time.time()
    try:
        # We run it and let it output to terminal so we can see progress
        process = subprocess.Popen(cmd)
        process.wait()
        end = time.time()
        print(f">>> FINISHED: {model} ({profile}) in {end-start:.1f}s")
    except Exception as e:
        print(f">>> ERROR running {model} ({profile}): {e}")

def main():
    # Attempt cleanup of results folder
    print(">>> Attempting to clear results/ folder...")
    # if results folder exists, try to delete subfolders
    if os.path.exists("results"):
        for item in os.listdir("results"):
            item_path = os.path.join("results", item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Cleared {item_path}")
                except Exception as e:
                    print(f"Could not clear {item_path}: {e}")

    # Run matrix
    for profile in PROFILES:
        for model in MODELS:
            run_sim(model, profile)

if __name__ == "__main__":
    main()
