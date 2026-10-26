# Mini GPT: A beginner's GPT-style language model from scratch

A small character-level, decoder-only Transformer implemented in PyTorch. **No pretrained model weights.** Designed for VS Code on an Apple Silicon Mac. Uses Apple's GPU via PyTorch MPS, or CPU fallback.

## 1. Open in VS Code

Clone or download this repository, then open **File > Open Folder** in VS Code and select the project folder.

Install Microsoft's **Python** extension if prompted.

## 2. Run in the VS Code Terminal

Install Python 3.12 first. Open **Terminal > New Terminal**, then enter the following commands **one by one**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_data.py
python mini_gpt.py --steps 500
```

The last command trains the model from random initialization, saves `mini_gpt.pt` and prints generated text. The first dataset download needs an internet connection; training does not. If the Mac GPU is available, the output starts with `Using device: mps`.

## 3. Generate again without retraining

```bash
python mini_gpt.py --generate-only --prompt "ROMEO:" --tokens 250
```

The prompt must use characters present in the training dataset (for example, Shakespeare text has upper-case names). If you get a `No mini_gpt.pt found` error, train first.

## 4. Experiment

- Begin with `--steps 50` to test the entire pipeline.
- Train for more iterations with `--steps 1500` to study text quality and validation loss. Each new training command initializes a **new** model from scratch; it does not resume training.
- Change `NUM_LAYERS`, `EMBED_SIZE`, `BLOCK_SIZE` in `mini_gpt.py` to explore model capacity (but this increases compute and memory requirements).
- Replace `input.txt` with a different sufficiently large UTF-8 text collection to train on a different style. This creates a **new** tokenizer and requires retraining.

## 5. What the code is doing

1. Read the text and map each unique character to an integer.
2. Sample batches of 128-character text windows and their shifted next-character targets.
3. Convert character IDs to learned token and position embeddings.
4. Use masked multi-head self-attention and feed-forward networks to predict next characters.
5. Compute cross-entropy loss against the true next characters.
6. Backpropagate the loss and update **all** model parameters using AdamW.
7. Save a checkpoint (weights, vocabulary, model configuration), then generate new characters one at a time.

## 6. Common issues

- `python3: command not found`: install a compatible ARM64 Python, preferably Python 3.11 or 3.12, then create the virtual environment again.
- `ModuleNotFoundError: torch`: activate `.venv` and install requirements.
- `Using device: cpu`: PyTorch's MPS backend is unavailable in this Python setup. Training still works on CPU but may be slower.
- `input.txt is missing`: run `python download_data.py` while connected to the internet.
- `Prompt contains characters missing from the dataset`: use a prompt that only includes characters found in the text; e.g. `ROMEO:`.
- Mac memory pressure: decrease `BATCH_SIZE` to 4 or 8, or decrease `BLOCK_SIZE` to 64.

## 7. Notes

This is a real language model trained from scratch, but a **small educational GPT-style model**, not a full-scale conversational assistant. The tiny dataset and character-level tokenizer limit its text quality. It requires no Hugging Face access, API key, or cloud GPU.

Dataset: [Andrej Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). Please review its source and terms before redistributing training text.

## 8. Files kept locally

Git excludes the downloaded `input.txt`, trained `*.pt` checkpoints, virtual environments, caches, `.env` files, and machine-specific VS Code settings. Run the setup and training commands above after cloning to recreate the data and model.
