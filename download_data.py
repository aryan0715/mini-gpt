"""Download a small text dataset once (internet required)."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DESTINATION = Path("input.txt")

if DESTINATION.exists():
    print("input.txt already exists. Not overwriting it.")
else:
    print("Downloading Tiny Shakespeare...")
    with urlopen(URL, timeout=30) as response:
        content = response.read().decode("utf-8")
    DESTINATION.write_text(content, encoding="utf-8")
    print(f"Saved {len(content):,} characters to {DESTINATION}")
