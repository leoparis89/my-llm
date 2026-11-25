import tiktoken
import torch
import torch.nn as nn

# The final machine, built one piece at a time.
# Names follow the book's final model (chapter 4, GPTModel), so you can look anything up,
# and so OpenAI's trained GPT-2 numbers can be loaded into it at the end.

# Tiny sizes for learning, the same as ch02.py: cards of 4 tokens, 3 numbers per token.
GPT_CONFIG_TINY = {
    "vocab_size": 50257,  # tokens in GPT-2's list (fixed by the tokenizer, can't shrink)
    "context_length": 4,  # biggest card: 4 tokens
    "emb_dim": 3,  # numbers per token
    "n_heads": 1,  # attention heads (used later)
    "n_layers": 1,  # blocks stacked on top of each other (used later)
    "drop_rate": 0.1,  # dropout: share of numbers zeroed during training (used later)
    "qkv_bias": False,  # whether the query/key/value filters add an extra number (used later)
}

# Real GPT-2 small sizes (124 million learned numbers). We switch to these once everything works.
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# A nn.Module is PyTorch's box for a model:
#   __init__ creates the tables and filters (all random at first, adjusted by training)
#   forward  is what happens when you call model(token_ids)
# Sizes in the comments below are for GPT_CONFIG_TINY.
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Word table: (50,257 x 3), one row of 3 numbers per token (same as embedding_layer in ch02.py)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Position table: (4 x 3), one row of 3 numbers per spot (same as position_layer in ch02.py)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dictionary table for guessing: one row of 3 random numbers for each of the 50,257 dictionary words.
        #   size (50,257 x 3)
        #   row 262   ' the'   [-0.38,  0.23, -0.38]
        #   row 530   ' one'   [-0.46,  0.49, -0.56]
        #   row 6549  ' surve' [ 0.58, -0.57, -0.54]
        #   ... 50,257 rows
        # These are NOT scores. The scores come out in forward, when a card word is multiplied with each row.
        # It's a separate table from tok_emb above: same size, different numbers, different job.
        #   tok_emb  gives each word ON THE CARD its 3 numbers (at the start)
        #   out_head gives each DICTIONARY WORD 3 numbers, used to score it as the next word (at the end)
        # How it's used is explained in forward, at the logits line.
        # nn.Linear(3, 50257) means "3 numbers in, 50,257 scores out". It creates this (50,257 x 3) table
        # and, when called, does the multiplying for us.
        # bias=False: just the multiplying, no extra number added afterwards.
        # Training adjusts these numbers so the right next word gets the highest score.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # token_ids = a grid of token IDs: one row per card, 4 IDs per row. (The book calls it in_idx.)
    # You can feed 1 card or many cards at once; each row is handled the same way.
    #   1 card (what we do below):
    #     [[7120, 7002, 4940, 351]]                "Your journey starts with"
    #   2 cards (what training will do, with many more):
    #     [[7120, 7002, 4940, 351],                "Your journey starts with"
    #      [7002, 4940,  351, 530]]                "journey starts with one"
    def forward(self, token_ids):
        # How many cards, and how many words per card: here 1 card of 4 words.
        batch_size, seq_len = token_ids.shape  # batch_size = 1, seq_len = 4

        # Each word's ID -> its row in the word table (3 numbers per word):
        #   Your    (7120) -> [ 0.67,  0.25, -1.66]
        #   journey (7002) -> [ 1.05,  0.36, -0.70]
        #   starts  (4940) -> [ 2.53,  0.34, -0.25]
        #   with    (351)  -> [ 0.32, -0.85, -1.47]
        tok_embeds = self.tok_emb(token_ids)

        # Each spot on the card -> its row in the position table (3 numbers per spot).
        # torch.arange(4) = [0, 1, 2, 3], one number per spot.
        # (device=... just keeps the numbers on the same chip as token_ids: CPU or GPU.)
        #   spot 0 -> [-1.99,  0.45, -0.71]
        #   spot 1 -> [-0.05,  0.03,  0.30]
        #   spot 2 -> [ 0.36, -0.28, -0.19]
        #   spot 3 -> [ 1.62, -0.40,  0.04]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=token_ids.device))

        # Add them: word numbers + spot numbers, so the same word in another spot gets different numbers.
        #   Your    [ 0.67,  0.25, -1.66] + spot 0 [-1.99,  0.45, -0.71] = [-1.32,  0.70, -2.37]
        #   journey [ 1.05,  0.36, -0.70] + spot 1 [-0.05,  0.03,  0.30] = [ 1.00,  0.38, -0.40]
        #   starts  [ 2.53,  0.34, -0.25] + spot 2 [ 0.36, -0.28, -0.19] = [ 2.89,  0.06, -0.45]
        #   with    [ 0.32, -0.85, -1.47] + spot 3 [ 1.62, -0.40,  0.04] = [ 1.94, -1.26, -1.43]
        x = tok_embeds + pos_embeds  # (4 words x 3)

        # Attention and the rest of GPT-2 will go here, in later steps.

        # Score every dictionary word as the next word.
        # For each of our 4 words, multiply its 3 numbers (from x above)
        # with the 3 numbers of each of the 50,257 dictionary words (from out_head).
        # Each multiplication gives one score: how likely that dictionary word comes next.
        #
        # Example: how likely is ' one' after "with"?
        #   with's 3 numbers (from x):          [ 1.94, -1.26, -1.43]
        #   ' one''s 3 numbers (from out_head): [-0.46,  0.49, -0.56]
        #   multiply pair by pair and add:  1.94 x -0.46  +  -1.26 x 0.49  +  -1.43 x -0.56  =  -0.70
        #   -> score of ' one' after "with" = -0.70
        #
        # Same for all 50,257 dictionary words -> 50,257 scores for "with":
        #   ' the' -0.48   ' one' -0.70   ' surve' 2.62 (highest = the guess)   ...
        # Same for Your, journey and starts -> 50,257 scores each.
        #
        # Sizes: x (4 words x 3) with out_head (50,257 rows x 3)
        #        -> each of the 4 words gets a score from each of the 50,257 rows -> logits (4 words x 50,257)
        # (With more cards, the same happens for every card: (cards x 4 words x 50,257).)
        # "logits" is the official name for these scores.
        #
        # You don't see the multiplying in the line below because it's hidden inside it:
        # calling self.out_head(x) means "multiply x with my table".
        # It does NOT just return out_head's random numbers: different x -> different scores.
        # Written out by hand, it's the same as:
        #   logits = x @ self.out_head.weight.T
        #   (self.out_head.weight = the (50,257 x 3) table, .T = flipped, @ = the multiplying)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)  # same random numbers every run

# Create the machine: an object (instance) of the GPTModel class.
# This runs __init__ above with cfg = GPT_CONFIG_TINY, which builds tok_emb, pos_emb and out_head
# filled with random numbers.
model = GPTModel(GPT_CONFIG_TINY)

tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2's tokenizer, same as ch02.py

# Text -> token IDs: [7120, 7002, 4940, 351]
# The extra [ ] around it makes a batch of 1 card: [[7120, 7002, 4940, 351]], size (1 x 4).
# The model always expects a batch of cards, even if there's only one.
token_ids = torch.tensor([tokenizer.encode("Your journey starts with")])


# Write text: guess the next word, add it to the end, repeat.
# The text grows, but the model only reads the last 4 words each round (the card size):
#   round 1: "Your journey starts with"   -> ' surve'
#   round 2: "journey starts with surve"  -> ' Sta'
#   round 3: "starts with surve Sta"      -> ' Fri'
#   round 4: "with surve Sta Fri"         -> ' speaks'
#   round 5: "surve Sta Fri speaks"       -> ' surve'
length_of_dance = 20  # how many words to add

result = token_ids  # the text so far, as a grid of 1 row: [[7120, 7002, 4940, 351]]

for i in range(length_of_dance):
    # Last 4 IDs of the text (the sliding card). ":" = every card, "-4:" = last 4 IDs.
    #   round 2: [[7120, 7002, 4940, 351, 6549]] -> [[7002, 4940, 351, 6549]]
    card = result[:, -4:]
    logits = model(card)  # scores: 1 card x 4 words x 50,257
    prediction_ids = logits.argmax(dim=-1)  # guess after each word: [[29817, 9209, 39683, 6549]]
    # Keep only the guess after the last word. "-1:" (with the colon) keeps the brackets: [[6549]]
    last_prediction_id = prediction_ids[:, -1:]
    # Add it to the end of the text. dim=1 = along the IDs (dim=0 would add a new card).
    #   [[7120, 7002, 4940, 351]] + [[6549]] -> [[7120, 7002, 4940, 351, 6549]]
    result = torch.cat((result, last_prediction_id), dim=1)

    # ids = prediction_ids[0].tolist()   # [[29817, 9209, 39683, 6549]] -> [29817, 9209, 39683, 6549]
    # for id in ids:
    #     print(id, tokenizer.decode([id]))   # decode the id to a word

    # print(f"prediction_ids: {prediction_ids}")
    # print(f"last_prediction_id: {last_prediction_id}")
    # print(f"current: {card}")

    # print(f"step {i}:")


print(f"result: {result}")
result_words = tokenizer.decode(result[0].tolist())
print(f"result_words: {result_words}")





