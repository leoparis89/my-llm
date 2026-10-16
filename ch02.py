import os
import urllib.request

import tiktoken
import torch

# Step 1: get the training text ("The Verdict" by Edith Wharton, public domain)
file_path = "the-verdict.txt"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
    "main/ch02/01_main-chapter-code/the-verdict.txt"
)

if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)

# Step 2: read it into one big string
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 3: look at it
print("Total number of characters:", len(raw_text))
print(raw_text[:99])

# Step 4: turn the text into token IDs with GPT-2's tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(raw_text)

print("\nTotal number of tokens:", len(token_ids))
print("First 10 token IDs:", token_ids[:10])

# Step 5: see which piece of text each ID stands for
print("\nID     -> text piece")
for token_id in token_ids[:10]:
    print(f"{token_id:<6} -> {tokenizer.decode([token_id])!r}")

# Step 6: decode back to text to check nothing was lost
print("\nRound trip matches original:", tokenizer.decode(token_ids) == raw_text)

# Step 7: the target is the input shifted by one token
x = token_ids[:4]
y = token_ids[1:5]
print("\nx:", x)
print("y:", y)

# Step 8: slide the window along the whole text
window = 4  # how many tokens the model sees at once
stride = 1  # how far the window moves each time
inputs = []
targets = []
for start in range(0, len(token_ids) - window, stride):
    inputs.append(token_ids[start : start + window])
    targets.append(token_ids[start + 1 : start + window + 1])

print("\nNumber of examples:", len(inputs))
for i in range(3):
    print(f"\nexample {i + 1}:")
    print("  x:", inputs[i], repr(tokenizer.decode(inputs[i])))
    print("  y:", targets[i], repr(tokenizer.decode(targets[i])))

# Step 9: stack the first 8 cards into tensors so they can be processed in parallel
batch_inputs = torch.tensor(inputs[:8])
batch_targets = torch.tensor(targets[:8])
print("\nBatch inputs shape:", batch_inputs.shape)
print(batch_inputs)
print("Batch targets shape:", batch_targets.shape)
print(batch_targets)

# Step 10: embeddings -- a list of 3 random numbers for each of GPT-2's 50,257 tokens
torch.manual_seed(123)  # same random numbers every run
embedding_layer = torch.nn.Embedding(50257, 3)

# Look up the embeddings for the 4 tokens on card 1
card = batch_inputs[0]
card_embeddings = embedding_layer(card)
print("\nEmbeddings for card 1:")
for token_id, embedding in zip(card.tolist(), card_embeddings.tolist()):
    numbers = ", ".join(f"{n:6.2f}" for n in embedding)
    print(f"{tokenizer.decode([token_id])!r:<10} (ID {token_id:<5}) -> [{numbers}]")
