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
        # Dictionary table for guessing: 3 random numbers for each of the 50,257 dictionary words.
        #   size (3 x 50,257)
        # It's a separate table from tok_emb above: same size, different numbers, different job.
        #   tok_emb  gives each word ON THE CARD its 3 numbers (at the start)
        #   out_head gives each DICTIONARY WORD 3 numbers, used to score it as the next word (at the end)
        # How it's used is explained in forward, at the logits line.
        # nn.Linear(3, 50257) creates this table and, when called, does the multiplying for us.
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
        # Sizes: x (4 words x 3) times out_head (3 x 50,257) -> logits (4 words x 50,257)
        # (With more cards, the same happens for every card: (cards x 4 words x 50,257).)
        # "logits" is the official name for these scores.
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

# Run the machine on the card.
# model is an object, but it can be called like a function: model(token_ids) runs forward(token_ids) above.
# That's built into nn.Module: its __call__ method calls forward.
# (See __call__ and _call_impl in .venv/lib/python3.12/site-packages/torch/nn/modules/module.py)
# logits: for each of the 4 words on the card, a score for each of the 50,257 candidates
# as the next word. Big score = the model thinks that candidate is likely to come next.
# A score's position in the list of 50,257 is the candidate's token ID, so decode turns it into the word.
# Example, the scores after "with" (card 0, spot 3):
#   logits[0, 3, 262]  = -0.48   tokenizer.decode([262])  = ' the'
#   logits[0, 3, 530]  = -0.70   tokenizer.decode([530])  = ' one'
#   logits[0, 3, 6549] =  2.62   tokenizer.decode([6549]) = ' surve'   <- biggest of all 50,257
logits = model(token_ids)  # (1 card x 4 tokens x 50,257)
print("Output size:", tuple(logits.shape))

# The guess at each spot = the candidate with the biggest score.
# argmax gives the POSITION of the biggest score (not the score itself) = that candidate's token ID.
#   after "with": biggest score is 2.62, at position 6549 -> guess = 6549 = ' surve'
# dim=-1 means "look along the last size" (the 50,257), so we get one ID per word.
# Like token_ids, it's a grid with one row per card. We have 1 card, so 1 row:
#   guesses = [[29817, 9209, 39683, 6549]] = ['VIS', ' speaks', 'ß', ' surve']
guesses = logits.argmax(dim=-1)

# Print, for each word on the card, the text read so far and the model's guess for the next word:
#   after 'Your' -> guess 'VIS'
#   after 'Your journey' -> guess ' speaks'
#   after 'Your journey starts' -> guess 'ß'
#   after 'Your journey starts with' -> guess ' surve'
#
# First, get both as plain lists of numbers:
#   [0]       takes row 0 out of the grid (our first and only card):
#               [[7120, 7002, 4940, 351]]  ->  [7120, 7002, 4940, 351]
#   .tolist() turns PyTorch's tensor into a normal Python list, which tokenizer.decode can read:
#               tensor([7120, 7002, 4940, 351])  ->  [7120, 7002, 4940, 351]
card_ids = token_ids[0].tolist()  # the card's words:    [7120, 7002, 4940, 351]
guess_ids = guesses[0].tolist()  # the model's guesses: [29817, 9209, 39683, 6549]
text_so_far = ""
for spot in range(4):  # spot = 0, 1, 2, 3
    word = tokenizer.decode([card_ids[spot]])  # spot 1: 7002 -> ' journey'
    text_so_far = text_so_far + word  # spot 1: 'Your' + ' journey' -> 'Your journey'
    guess = tokenizer.decode([guess_ids[spot]])  # spot 1: 9209 -> ' speaks'
    print(f"after '{text_so_far}' -> guess '{guess}'")
