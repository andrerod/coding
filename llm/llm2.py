import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Geography dataset
# ============================================================

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

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Use CPU for compatibility with your MacBook Air.
device = torch.device("cpu")


# ============================================================
# 2. Create question-answer examples
# ============================================================


def make_question(subject):
    if subject.startswith("capital of "):
        country = subject.removeprefix("capital of ")
        return f"What is the capital of {country}?"

    if subject.startswith("continent containing "):
        country = subject.removeprefix("continent containing ")
        return f"Which continent is {country} in?"

    return f"What is the {subject}?"


examples = []

for fact_id, (subject, answer) in enumerate(facts):
    question = make_question(subject)

    examples.append(
        {
            "fact_id": fact_id,
            "prompt": f"Question: {question}\nAnswer: ",
            "answer": answer,
        }
    )


# Hold out entire facts to reduce data leakage.
fact_ids = list(range(len(facts)))
random.shuffle(fact_ids)

n_val = max(1, round(0.2 * len(fact_ids)))
val_ids = set(fact_ids[:n_val])

train_examples = [ex for ex in examples if ex["fact_id"] not in val_ids]

val_examples = [ex for ex in examples if ex["fact_id"] in val_ids]

print("Training facts:", len(train_examples))
print("Validation facts:", len(val_examples))


# ============================================================
# 3. Character tokenizer + dedicated END token
# ============================================================

END_TOKEN = "<END>"

all_text = "".join(ex["prompt"] + ex["answer"] for ex in examples)

chars = sorted(set(all_text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

END_ID = len(chars)
vocab_size = len(chars) + 1


def encode(text):
    unknown = set(text) - set(stoi)

    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    return [stoi[ch] for ch in text]


def decode(ids):
    # END_ID is a control token, not a printable character.
    return "".join(itos[token_id] for token_id in ids if token_id != END_ID)


print("Vocabulary size:", vocab_size)


# ============================================================
# 4. Prepare training sequences and answer-only labels
# ============================================================

block_size = 256
batch_size = 8


def encode_example(example):
    prompt_ids = encode(example["prompt"])
    answer_ids = encode(example["answer"])

    # The model learns to predict the answer and then END_ID.
    ids = prompt_ids + answer_ids + [END_ID]

    # Inputs predict the next token at each position.
    x = ids[:-1]

    # -100 tells cross_entropy to ignore prompt positions.
    y = [-100] * len(x)

    answer_start = len(prompt_ids)

    # Predict the first answer character from the final
    # prompt character, then predict the remaining answer
    # characters and END_ID.
    for pos in range(answer_start - 1, len(x)):
        y[pos] = ids[pos + 1]

    if len(x) > block_size:
        raise ValueError(f"Example length {len(x)} exceeds block_size={block_size}")

    return x, y


def make_batch(batch_examples):
    encoded = [encode_example(ex) for ex in batch_examples]

    max_len = max(len(x) for x, _ in encoded)

    # Right-pad input sequences with token ID 0.
    x_batch = torch.zeros(len(encoded), max_len, dtype=torch.long)

    # Ignore padded positions in the loss.
    y_batch = torch.full(
        (len(encoded), max_len),
        -100,
        dtype=torch.long,
    )

    for i, (x, y) in enumerate(encoded):
        x_batch[i, : len(x)] = torch.tensor(x)
        y_batch[i, : len(y)] = torch.tensor(y)

    return x_batch.to(device), y_batch.to(device)


# ============================================================
# 5. Token and position embeddings
# ============================================================


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()

        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device)

        return self.token(idx) + self.position(positions)


# ============================================================
# 6. Transformer block with causal self-attention
# ============================================================


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attn = nn.MultiheadAttention(
            n_embd,
            n_heads,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        _, T, _ = x.shape

        # True means the position cannot be attended to.
        # Prevent the model from looking at future tokens.
        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.ln1(x)

        attention_output, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False,
        )

        # Residual connection.
        x = x + attention_output

        # Feed-forward network and another residual.
        x = x + self.ffn(self.ln2(x))

        return x


# ============================================================
# 7. GPT-style language model
# ============================================================


class GeographyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=128,
        n_heads=4,
        n_layers=2,
        block_size=256,
    ):
        super().__init__()

        self.block_size = block_size

        self.embedding = TokenAndPositionEmbedding(
            vocab_size,
            n_embd,
            block_size,
        )

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
                ignore_index=-100,
            )

        return logits, loss


# ============================================================
# 8. Initialize model and optimizer
# ============================================================

model = GeographyGPT(
    vocab_size=vocab_size,
    n_embd=128,
    n_heads=4,
    n_layers=2,
    block_size=block_size,
).to(device)

num_parameters = sum(p.numel() for p in model.parameters())

print(f"Model parameters: {num_parameters:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)


# ============================================================
# 9. Evaluate training and validation loss
# ============================================================


@torch.no_grad()
def evaluate(examples):
    model.eval()

    losses = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]

        x, y = make_batch(batch)
        _, loss = model(x, y)

        if loss is not None:
            losses.append(loss.item())

    model.train()

    if not losses:
        return float("nan")

    return sum(losses) / len(losses)


# ============================================================
# 10. Train from random initialization
# ============================================================

steps = 3000

for step in range(steps):
    model.train()

    batch = random.sample(
        train_examples,
        min(batch_size, len(train_examples)),
    )

    x, y = make_batch(batch)

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step % 100 == 0 or step == steps - 1:
        train_loss = evaluate(train_examples)
        val_loss = evaluate(val_examples)

        print(
            f"step={step:4d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )


# ============================================================
# 11. Save model, vocabulary, and configuration
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "end_id": END_ID,
    "vocab_size": vocab_size,
    "config": {
        "n_embd": 128,
        "n_heads": 4,
        "n_layers": 2,
        "block_size": block_size,
    },
}

torch.save(checkpoint, "geography_gpt_v2.pt")

print("Saved geography_gpt_v2.pt")


# ============================================================
# 12. Generate an answer
# ============================================================


@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.0,
):
    model.eval()

    # Encode the prompt, preserving the complete context.
    idx = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size :]

        logits, _ = model(context)

        # Predict from the final position only.
        logits = logits[:, -1, :]

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(
                logits / temperature,
                dim=-1,
            )

            next_id = torch.multinomial(probs, num_samples=1)

        token_id = next_id.item()

        # Stop on the dedicated end token.
        if token_id == END_ID:
            break

        generated_ids.append(token_id)

        idx = torch.cat(
            [idx, next_id],
            dim=1,
        )

    return decode(generated_ids)


# ============================================================
# 13. Test the trained model
# ============================================================

tests = [
    "Question: What is the capital of France?\nAnswer: ",
    "Question: What is the capital of Japan?\nAnswer: ",
    "Question: Which continent is Brazil in?\nAnswer: ",
    "Question: What is the largest ocean?\nAnswer: ",
]

print("\n--- Generation tests ---")

for prompt in tests:
    answer = generate(
        model,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
    )

    print("\nQuestion:", prompt.split("Question: ")[1].split("\n")[0])
    print("Generated answer:", repr(answer))


# ============================================================
# 14. Inspect held-out examples
# ============================================================

print("\n--- Held-out validation examples ---")

for ex in val_examples:
    answer = generate(
        model,
        ex["prompt"],
        max_new_tokens=100,
        temperature=0.0,
    )

    print("Question:", ex["prompt"].split("Question: ")[1].split("\n")[0])
    print("Expected:", ex["answer"])
    print("Predicted:", repr(answer))
    print()


def answer_question(question):
    question = question.strip()

    # Let the user type either a plain question or a full prompt.
    if question.startswith("Question:"):
        prompt = question
        if "Answer:" not in prompt:
            prompt += "\nAnswer: "
    else:
        prompt = f"Question: {question}\nAnswer: "

    return generate(
        model,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
    )


while True:
    question = input("\nAsk a geography question (or type 'exit'): ")

    if question.strip().lower() == "exit":
        break

    print("Answer:", repr(answer_question(question)))
