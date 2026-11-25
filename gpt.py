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

    # token_ids = one card: the IDs of up to 4 words. (The book calls it in_idx.)
    #   [7120, 7002, 4940, 351]   "Your journey starts with"
    # (The book feeds many cards at once, as a grid. We'll do that later, when training on many cards.)
    def forward(self, token_ids):
        # How many words on the card: 4
        seq_len = len(token_ids)

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


# Write text (generation): guess the next word, add it to the end, repeat.
# This is how ChatGPT writes its answers, one word at a time. (The book calls it generate_text_simple.)
# The text grows, but the model only reads the last 4 words each round (the card size):
#   round 1: "Your journey starts with"   -> ' surve'
#   round 2: "journey starts with surve"  -> ' Sta'
#   round 3: "starts with surve Sta"      -> ' Fri'
#   round 4: "with surve Sta Fri"         -> ' speaks'
#   round 5: "surve Sta Fri speaks"       -> ' surve'
#
#   starting_token_ids = the starting text, e.g. [7120, 7002, 4940, 351] = "Your journey starts with"
#   context_size       = biggest card the model can read (4 for GPT_CONFIG_TINY, 1,024 for real GPT-2)
#   length_of_dance    = how many words to add
# Returns the starting text with the new words added at the end.
def generate_text(model, starting_token_ids, context_size, length_of_dance=20):
    result = starting_token_ids  # the text so far: [7120, 7002, 4940, 351]
    for _i in range(length_of_dance):
        # Last 4 IDs of the text (the sliding card). "-4:" = the last 4.
        #   round 2: [7120, 7002, 4940, 351, 6549] -> [7002, 4940, 351, 6549]
        card = result[-context_size:]
        logits = model(card)  # scores: 4 words x 50,257
        prediction_ids = logits.argmax(dim=-1)  # guess after each word: [29817, 9209, 39683, 6549]
        # Keep only the guess after the last word. "-1:" = the last one: [6549]
        last_prediction_id = prediction_ids[-1:]
        # Add it to the end of the text.
        #   [7120, 7002, 4940, 351] + [6549] -> [7120, 7002, 4940, 351, 6549]
        result = torch.cat((result, last_prediction_id))
    return result


torch.manual_seed(123)  # same random numbers every run

# Create the machine: an object (instance) of the GPTModel class.
# This runs __init__ above with cfg = GPT_CONFIG_TINY, which builds tok_emb, pos_emb and out_head
# filled with random numbers.
model = GPTModel(GPT_CONFIG_TINY)

tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2's tokenizer, same as ch02.py

# Text -> token IDs: [7120, 7002, 4940, 351, 257]
token_ids = torch.tensor(tokenizer.encode("Your journey starts with a"))


# How wrong is the model on a card? One number: big = very wrong, close to 0 = right.
#   card          = [7120, 7002, 4940, 351]   "Your journey starts with"
#   right_answers = [7002, 4940, 351, 257]    "journey starts with a"  (the card slid by 1)
def calc_loss(model, card, right_answers):
    logits = model(card)  # scores: 4 words x 50,257
    # Compare the 4 guesses' scores with the 4 right answers.
    loss = nn.CrossEntropyLoss()(logits, right_answers)
    return loss


# Cut the card and the right answers out of the 5-word text:
#   text:           Your  journey  starts  with  a     [7120, 7002, 4940, 351, 257]
#   card:           Your  journey  starts  with        words 1-4: [:4]
#   right answers:        journey  starts  with  a     words 2-5: [1:]
# ":4" = from the start up to position 4 (not included), "1:" = from position 1 to the end.
card = token_ids[:4]
right_answers = token_ids[1:]

loss = calc_loss(model, card, right_answers)
print(f"loss: {loss.item():.3f}")  # .item() turns the 1-number tensor into a plain number

