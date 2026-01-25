from comet import download_model
import os

print("Starting setup_comet.py...", flush=True)

# Attempt to download/load model explicitly
try:
    print("Downloading/Checking wmt22-cometkiwi-da...", flush=True)
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    print(f"Model path: {model_path}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

print("Done.", flush=True)
