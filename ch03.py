import torch

# Chapter 3, simple attention: the book's example sentence, with its hand-picked scores.
# A real model learns these scores; here they're fixed so the output matches the book.
words = ["Your", "journey", "starts", "with", "one", "step"]
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)

# Part 1: how similar is "journey" to each word? (the book calls these "attention scores")
# Similarity = dot product. Bigger = more similar.
# From scratch, torch.dot(a, b) is: sum(x * y for x, y in zip(a, b))
query = inputs[1]  # "journey"
attn_scores = torch.tensor([torch.dot(query, word_scores) for word_scores in inputs])

for word, score in zip(words, attn_scores):
    print(f"journey vs {word:<8} {score:.4f}")

# Part 2: turn the scores into shares that add up to 1 (100%).
# The book calls these "attention weights".

# Simple way: divide each score by the total.
simple_weights = attn_scores / attn_scores.sum()

# The way actually used: softmax. Same idea, but it makes big scores count even more,
# and it always gives positive shares, even if some scores are negative.
# From scratch, torch.softmax(x, dim=0) is: torch.exp(x) / torch.exp(x).sum()
attn_weights = torch.softmax(attn_scores, dim=0)

print("\nShare of each word:  simple   softmax")
for word, simple, soft in zip(words, simple_weights, attn_weights):
    print(f"  {word:<8}           {simple:.4f}   {soft:.4f}")
print(f"  {'total':<8}           {simple_weights.sum():.4f}   {attn_weights.sum():.4f}")
