import torch
import torch.nn as nn
import torch.nn.functional as F
from llm import decode, encode, model

# Load the learned parameters into the model architecture.
state_dict = torch.load("geography_gpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

model.eval()


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


while True:
    prompt = input("Prompt: ")
    if prompt == "exit":
        break

    print(generate(model, prompt))
