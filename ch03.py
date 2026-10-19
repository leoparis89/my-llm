import torch

# Chapter 3, simple attention: the book's example sentence.
words = ["Your", "journey", "starts", "with", "one", "step"]

# Embeddings: 3 numbers describing each word.
# In the real model, each row is what chapter 2 produces (card_with_positions in ch02.py):
#   word -> token ID -> word scores (word embedding) + spot scores (position embedding)
# The book skips those steps and writes down made-up results directly,
# so the attention math is easy to follow and the output matches the book.
# Chapter 4 connects the two: chapter 2's output goes straight into attention.
# (The book calls this variable "inputs", since it's the input to attention.)
embeddings = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)

# Part 1: how similar is "journey" to each word?
# Similarity = dot product of the two embeddings. Bigger = more similar.
# From scratch, torch.dot(a, b) is: sum(x * y for x, y in zip(a, b))
# (The book calls the word doing the comparing the "query",
# and these similarities "attention scores".)
query = embeddings[1]  # "journey"
similarities = torch.tensor([torch.dot(query, embedding) for embedding in embeddings])

for word, similarity in zip(words, similarities):
    print(f"journey vs {word:<8} {similarity:.4f}")

# Part 2: turn the similarities into shares that add up to 1 (100%).
# (The book calls these shares "attention weights".)

# Simple way: divide each similarity by the total.
simple_shares = similarities / similarities.sum()

# The way actually used: softmax. Same idea, but bigger similarities get an even bigger
# share, and every share is positive, even if some similarities are negative.
# From scratch, torch.softmax(x, dim=0) is: torch.exp(x) / torch.exp(x).sum()
shares = torch.softmax(similarities, dim=0)

print("\nShare of each word:  simple   softmax")
for word, simple, soft in zip(words, simple_shares, shares):
    print(f"  {word:<8}           {simple:.4f}   {soft:.4f}")
print(f"  {'total':<8}           {simple_shares.sum():.4f}   {shares.sum():.4f}")
