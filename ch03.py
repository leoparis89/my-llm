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
#
# Sizes are written as (rows x columns).
# `.T` flips rows and columns: a (6 x 3) grid becomes (3 x 6).
# `@` is matrix multiplication: each cell of the result = dot product of
#   a row of the left grid with a column of the right grid.
#   (a x b) @ (b x c) -> (a x c). The middle sizes must match; the outer sizes give the result.
#
# embeddings:    (6 x 3)  6 rows = words, 3 columns = numbers per word
# embeddings.T:  (3 x 6)  6 columns = words, each word's 3 numbers now going down a column

# 1. Attention scores for every pair of words.
#    (6 x 3) @ (3 x 6) -> (6 x 6)
#    Row i = word i asking, column j = word j being compared with.
#    Cell (i, j) = dot product of word i with word j.
#    Row 2 = the "journey" scores from part 1.
all_attention_scores = embeddings @ embeddings.T  # (6 x 6)

# 2. Softmax on each row, so each word's shares add up to 1. The size doesn't change.
#    dim=1 means "along each row" (across the columns).
#    Row i = how much of word i's blend comes from each of the 6 words.
#    Row 2 = the "journey" shares from part 2.
all_shares = torch.softmax(all_attention_scores, dim=1)  # (6 x 6)

# 3. Blend.
#    (6 x 6) @ (6 x 3) -> (6 x 3)
#    Cell (i, k) = row i of all_shares (word i's 6 shares)
#                  dotted with column k of embeddings (the k-th number of all 6 words)
#                = share_1 * word1[k] + share_2 * word2[k] + ... + share_6 * word6[k]
#    That's exactly the loop in part 3, for every word and every number at once.
#    Row i = new version of word i (3 numbers). Row 2 = the new "journey" from part 3.
all_new_embeddings = all_shares @ embeddings  # (6 x 3)

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

# Part 5: filters -- the part of attention that can learn.
# Until now, the same embedding was used in 3 different places. Example with journey and starts:
#   part 1:  score = dot(journey, starts)   <- journey on the left side, starts on the right side
#   part 3:  blend += 23% * starts          <- starts as the thing being mixed in
# So every embedding plays 3 roles: left side of a dot, right side of a dot, thing mixed in.
# One set of 3 numbers has to do all 3, and none of it can be learned.
# Now each word passes through 3 filters first, giving it a separate version for each role:
#   filter A -> left side of the dot   = "what I'm looking for"     (book: query)
#   filter B -> right side of the dot  = "what I can be matched on" (book: key)
#   filter C -> the thing mixed in     = "what I add to the blend"  (book: value) -- used in the next step
# Each filter is a (3 x 2) grid: it turns a word's 3 numbers into 2 numbers.
# The filters are random for now. Training will adjust them; that's how attention learns.
torch.manual_seed(123)  # the book's seed, so we get the book's numbers
filter_looking_for = torch.rand(3, 2)  # (3 x 2)  book: W_query
filter_matched_on = torch.rand(3, 2)  # (3 x 2)  book: W_key
filter_adds = torch.rand(3, 2)  # (3 x 2)  book: W_value

# Pass "journey" through filter A.
# (3) @ (3 x 2) -> (2): journey's 3 numbers become 2 numbers.
journey_looking_for = embeddings[1] @ filter_looking_for  # (2)  book: query_2

# Pass every word through filter B.
# (6 x 3) @ (3 x 2) -> (6 x 2): each of the 6 words' 3 numbers become 2 numbers.
all_matched_on = embeddings @ filter_matched_on  # (6 x 2)  book: keys

# Attention scores, same idea as part 1, but comparing the filtered versions:
# journey's "looking for" dotted with each word's "matched on".
# (2) @ (2 x 6) -> (6): one score per word.
filtered_scores = journey_looking_for @ all_matched_on.T  # (6)

print("\nWith filters, journey's attention scores:")
for word, score in zip(words, filtered_scores):
    print(f"journey vs {word:<8} {score:.4f}")
