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
#   forward  is what happens when you call model(in_idx)
# Sizes in the comments below are for GPT_CONFIG_TINY.
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Word table: (50,257 x 3), one row of 3 numbers per token (same as embedding_layer in ch02.py)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Position table: (4 x 3), one row of 3 numbers per spot (same as position_layer in ch02.py)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Guess table: (50,257 x 3), one row of 3 numbers per token in GPT-2's list. Random at first.
        # To score how likely a token in the list comes next: dot product of our token's 3 numbers
        # with that token's row (same as the attention scores in ch03.py). Example, after "with":
        #   with [1.94, -1.26, -1.43] . ' surve' row [0.58, -0.57, -0.54] =  2.62
        #   with [1.94, -1.26, -1.43] . ' one'   row [-0.46, 0.49, -0.56] = -0.70
        # Done for all 50,257 rows -> 50,257 numbers per token on the card (the logits). Biggest = the guess.
        # Training adjusts the rows so the right next word gets the biggest number.
        # nn.Linear creates the random table and does all the dot products at once with @:
        #   x (4 tokens x 3) @ table flipped (3 x 50,257) -> logits (4 tokens x 50,257)
        # bias=False: just the dot products, no extra number added afterwards.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):  # in_idx = the token IDs, one row per card
        batch_size, seq_len = in_idx.shape  # (cards, tokens per card)
        tok_embeds = self.tok_emb(in_idx)  # each token ID -> its row in the word table
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # spots 0, 1, 2, 3 -> their rows
        x = tok_embeds + pos_embeds  # (4 tokens x 3)
        # Attention and the rest of GPT-2 will go here, in later steps.
        logits = self.out_head(x)  # (4 tokens x 50,257)
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
in_idx = torch.tensor([tokenizer.encode("Your journey starts with")])

# Run the machine on the card.
# model is an object, but it can be called like a function: model(in_idx) runs forward(in_idx) above.
# That's built into nn.Module: its __call__ method calls forward.
# (See __call__ and _call_impl in .venv/lib/python3.12/site-packages/torch/nn/modules/module.py)
# logits: for each of the 4 words on the card, a score for each of the 50,257 candidates
# as the next word. Big score = the model thinks that candidate is likely to come next.
# A score's position in the list of 50,257 is the candidate's token ID, so decode turns it into the word.
# Example, the scores after "with" (card 0, spot 3):
#   logits[0, 3, 262]  = -0.48   tokenizer.decode([262])  = ' the'
#   logits[0, 3, 530]  = -0.70   tokenizer.decode([530])  = ' one'
#   logits[0, 3, 6549] =  2.62   tokenizer.decode([6549]) = ' surve'   <- biggest of all 50,257
logits = model(in_idx)  # (1 card x 4 tokens x 50,257)
print("Output size:", tuple(logits.shape))

# The guess at each spot = the candidate with the biggest score.
# argmax gives the POSITION of the biggest score (not the score itself) = that candidate's token ID.
#   after "with": biggest score is 2.62, at position 6549 -> guess = 6549 = ' surve'
# dim=-1 means "look along the last size" (the 50,257), so we get one ID per spot: size (1 x 4).
# [0] takes the first (and only) card: size (4).
#   guesses = [29817, 9209, 39683, 6549] = ['VIS', ' speaks', 'ß', ' surve']
guesses = logits.argmax(dim=-1)[0]

# Print what the model has seen so far at each spot, and its guess for the next token.
for spot, guess in enumerate(guesses.tolist()):  # spot = 0, 1, 2, 3
    seen = tokenizer.decode(in_idx[0, : spot + 1].tolist())  # the tokens up to and including this spot
    print(f"after {seen!r:<28} guess {tokenizer.decode([guess])!r}")
