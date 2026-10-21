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

# Part 1: attention scores -- how much should "journey" pay attention to each word?
# Score = dot product of the two embeddings. Bigger = pay more attention.
# The dot product is only a rough similarity: it also grows with how big the numbers are,
# so a word compared with itself is not always the highest, and there's no "perfect = 1".
# (Cosine similarity removes the size effect, but attention uses the plain dot product.)
# From scratch, torch.dot(a, b) is: sum(x * y for x, y in zip(a, b))
# (The book calls the word doing the comparing the "query".)
query = embeddings[1]  # "journey"
attention_scores = torch.tensor([torch.dot(query, embedding) for embedding in embeddings])

for word, score in zip(words, attention_scores):
    print(f"journey vs {word:<8} {score:.4f}")

# Part 2: turn the attention scores into shares that add up to 1 (100%).
# The shares are not similarities: they say how much of the blend in part 3 comes from each word.
# (The book calls these shares "attention weights".)

# Simple way: divide each score by the total.
simple_shares = attention_scores / attention_scores.sum()

# The way actually used: softmax. Same idea, but bigger scores get an even bigger share,
# and every share is positive, even if some scores are negative.
# From scratch, torch.softmax(x, dim=0) is: torch.exp(x) / torch.exp(x).sum()
shares = torch.softmax(attention_scores, dim=0)

print("\nShare of each word:  simple   softmax")
for word, simple, soft in zip(words, simple_shares, shares):
    print(f"  {word:<8}           {simple:.4f}   {soft:.4f}")
print(f"  {'total':<8}           {simple_shares.sum():.4f}   {shares.sum():.4f}")

# Part 3: build a new "journey" by blending all the words' embeddings, using the shares as the recipe.
# new journey = 14% of Your + 24% of journey + 23% of starts + ... (each word's 3 numbers × its share)
# "journey" gets the biggest share, but not 100%: the rest comes from the other words.
# That's how the new "journey" picks up information from the whole sentence.
# (The book calls the result the "context vector".)
new_journey = torch.zeros(3)  # empty total, filled by the loop below
for share, embedding in zip(shares, embeddings):
    new_journey += share * embedding

print("\nOld journey:", embeddings[1])
print("New journey:", new_journey)
