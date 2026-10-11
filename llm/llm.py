facts = [
    ("capital of France", "Paris"),
    ("capital of Japan", "Tokyo"),
    ("capital of Portugal", "Lisbon"),
    ("capital of Kenya", "Nairobi"),
    ("capital of Australia", "Canberra"),
    ("capital of Canada", "Ottawa"),
    ("capital of Brazil", "Brasilia"),
    ("capital of Egypt", "Cairo"),
    ("capital of Italy", "Rome"),
    ("capital of Peru", "Lima"),
    ("continent containing Brazil", "South America"),
    ("continent containing Egypt", "Africa"),
    ("continent containing Japan", "Asia"),
    ("continent containing Spain", "Europe"),
    ("largest ocean", "Pacific Ocean"),
    ("highest mountain above sea level", "Mount Everest"),
    (
        "longest river commonly cited in geography references",
        "The Nile or the Amazon, depending on the measurement method",
    ),
]

test_facts = [
    ("capital of Germany", "Berlin"),
    ("capital of Greece", "Athens"),
    ("capital of Mexico", "Mexico City"),
]

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        return self.token(idx) + self.position(positions)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        _, T, _ = x.shape

        # True means attention to that future position is forbidden.
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.ln1(x)
        a, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class GeographyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=128,
        n_heads=4,
        n_layers=2,
        block_size=128,
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding = TokenAndPositionEmbedding(vocab_size, n_embd, block_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T),
            )

        return logits, loss


block_size = 128
batch_size = 32


def get_batch(data):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


# 1. Build training examples
examples = []

for subject, answer in facts:
    if subject.startswith("capital of "):
        country = subject.removeprefix("capital of ")
        questions = [
            f"What is the capital of {country}?",
            f"Name the capital of {country}.",
            f"Which city is the capital of {country}?",
        ]
    elif subject.startswith("continent containing "):
        country = subject.removeprefix("continent containing ")
        questions = [
            f"Which continent is {country} in?",
            f"What continent contains {country}?",
        ]
    else:
        questions = [
            f"What is the {subject}?",
        ]

    for question in questions:
        examples.append(f"Question: {question}\n" f"Answer: {answer}.\n\n")

random.seed(42)
random.shuffle(examples)

text = "".join(examples)
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    unknown = set(s) - set(stoi)
    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)

print("Vocabulary size:", len(chars))
print("Training characters:", len(data))

model = GeographyGPT(
    vocab_size=len(chars),
    n_embd=128,
    n_heads=4,
    n_layers=2,
    block_size=block_size,
)


def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(3000):
        x, y = get_batch(data)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 300 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    torch.save(model.state_dict(), "geography_gpt.pt")


@torch.no_grad()
def generate(model, prompt, max_new_tokens=60, temperature=0.0):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    generated = ""

    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :]

        if temperature == 0:
            # Greedy decoding: select the most likely next character.
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        char = decode([next_id.item()])
        generated += char
        idx = torch.cat([idx, next_id], dim=1)

        if generated.endswith("\n\n"):
            break

    return generated


# tests = [
#     "Question: What is the capital of France?\nAnswer: ",
#     "Question: What is the capital of Japan?\nAnswer: ",
#     "Question: Which continent is Brazil in?\nAnswer: ",
#     "Question: What is the largest ocean?\nAnswer: ",
# ]


def main():
    train()

    while True:
        prompt = input("Prompt: ")
        if prompt == "exit":
            break

        print(generate(model, prompt))


if __name__ == "__main__":
    main()
