"""A small GPT-style language model trained entirely from scratch."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Edit these values later, after your first successful training run.
BATCH_SIZE = 16        # How many text examples to train on in a step
BLOCK_SIZE = 128       # How many previous characters the model can see
EMBED_SIZE = 128       # Size of each character's vector
NUM_HEADS = 4          # Number of attention heads
NUM_LAYERS = 4         # Transformer blocks
LEARNING_RATE = 3e-4


def choose_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.projection = nn.Linear(embed_size, embed_size)
        # Ones below the diagonal: tokens may look backward, not forward.
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch, time, channels = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def into_heads(tensor):
            return tensor.view(batch, time, self.num_heads, self.head_size).transpose(1, 2)

        query = into_heads(query)
        key = into_heads(key)
        value = into_heads(value)
        scores = query @ key.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = CausalSelfAttention(embed_size, num_heads, block_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocabulary_size, embed_size, num_heads, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocabulary_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.output = nn.Linear(embed_size, vocabulary_size)

    def forward(self, tokens, targets=None):
        _, sequence_length = tokens.shape
        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.norm(self.blocks(x))
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, count=300):
        self.eval()
        for _ in range(count):
            context = tokens[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / 0.8  # Temperature
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def main():
    parser = argparse.ArgumentParser(description="Train or use your own tiny GPT model.")
    parser.add_argument("--steps", type=int, default=500, help="Training steps (default 500)")
    parser.add_argument("--generate-only", action="store_true", help="Load trained checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Starting text for generation")
    parser.add_argument("--tokens", type=int, default=300, help="Characters to generate")
    args = parser.parse_args()
    if args.steps < 1 or args.tokens < 1:
        parser.error("--steps and --tokens must be positive numbers")

    torch.manual_seed(42)
    device = choose_device()
    print("Using device:", device)
    checkpoint_path = Path("mini_gpt.pt")

    if args.generate_only:
        if not checkpoint_path.exists():
            parser.error("No mini_gpt.pt found. Train first: python mini_gpt.py")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        chars = saved["chars"]
        config = saved["config"]
        model = MiniGPT(len(chars), **config)
        model.load_state_dict(saved["model_state"])
        model = model.to(device)
        print("Loaded trained model.")
    else:
        input_path = Path("input.txt")
        if not input_path.exists():
            parser.error("input.txt is missing. Run: python download_data.py")
        text = input_path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        char_to_id = {character: number for number, character in enumerate(chars)}
        token_ids = torch.tensor([char_to_id[ch] for ch in text], dtype=torch.long)
        cutoff = int(len(token_ids) * 0.9)
        training_tokens = token_ids[:cutoff]
        validation_tokens = token_ids[cutoff:]
        if min(len(training_tokens), len(validation_tokens)) <= BLOCK_SIZE + 1:
            parser.error("Training dataset is too short. Use a larger input.txt.")
        config = {
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
        }
        model = MiniGPT(len(chars), **config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("Vocabulary size:", len(chars))
        print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

        def get_batch(source):
            starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
            x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
            y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts])
            return x.to(device), y.to(device)

        for step in range(1, args.steps + 1):
            model.train()
            x, y = get_batch(training_tokens)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch(validation_tokens)
                    _, validation_loss = model(vx, vy)
                print(f"Step {step}/{args.steps} | train loss {loss.item():.3f} | validation loss {validation_loss.item():.3f}")

        torch.save({
            "model_state": model.cpu().state_dict(),
            "config": config,
            "chars": chars,
        }, checkpoint_path)
        model = model.to(device)
        print("Saved checkpoint:", checkpoint_path.resolve())

    char_to_id = {character: number for number, character in enumerate(chars)}
    starter = args.prompt or ("\n" if "\n" in char_to_id else chars[0])
    unknown = sorted(set(starter) - set(chars))
    if unknown:
        parser.error(f"Prompt contains characters missing from the dataset: {unknown}")
    seed = torch.tensor([[char_to_id[ch] for ch in starter]], device=device)
    generated_ids = model.generate(seed, count=args.tokens)[0].tolist()
    result = "".join(chars[token_id] for token_id in generated_ids)
    print("\n--- GENERATED TEXT ---\n" + result)


if __name__ == "__main__":
    main()
