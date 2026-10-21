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

# Part 4: do parts 1-3 for every word at once, so each word gets its own new version.
# `@` is matrix multiplication: it does many dot products in one go.
# `.T` flips rows and columns, so each word's embedding becomes a column to dot with.

# 1. Attention scores for every pair of words: a 6 x 6 table.
#    Row i, column j = dot product of word i with word j. Row 2 is the "journey" scores from part 1.
all_attention_scores = embeddings @ embeddings.T

# 2. Softmax on each row, so each word's shares add up to 1.
#    dim=1 means "along each row".
all_shares = torch.softmax(all_attention_scores, dim=1)

# 3. Blend: each word's row of shares × all the embeddings, added up.
#    Row 2 is the same new "journey" as in part 3.
all_new_embeddings = all_shares @ embeddings

header = "".join(f"{w:>9}" for w in words)
print("\nAttention scores (row = word doing the comparing):")
print(f"{'':<9}{header}")
for word, row in zip(words, all_attention_scores):
    print(f"{word:<9}" + "".join(f"{v:9.4f}" for v in row))

print("\nShares (each row adds up to 1):")
print(f"{'':<9}{header}")
for word, row in zip(words, all_shares):
    print(f"{word:<9}" + "".join(f"{v:9.4f}" for v in row))

print("\nNew embeddings for every word:")
for word, old, new in zip(words, embeddings, all_new_embeddings):
    print(f"  {word:<8} old {[round(v, 4) for v in old.tolist()]}  ->  new {[round(v, 4) for v in new.tolist()]}")

print("\nRow 2 matches part 3:", torch.allclose(all_new_embeddings[1], new_journey))
