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
        # Filter (a matrix, like the filters in ch03.py) that turns each token's 3 numbers
        # into 50,257 numbers: one per token in the list, saying how strongly the model thinks it comes next.
        #   x (4 tokens x 3) @ out_head (3 x 50,257) -> logits (4 tokens x 50,257)
        # nn.Linear creates the random (3 x 50,257) matrix and does the @ when called.
        # bias=False: just the @, no extra numbers added afterwards.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):  # in_idx = the token IDs, one row per card
        batch_size, seq_len = in_idx.shape  # (cards, tokens per card)
        tok_embeds = self.tok_emb(in_idx)  # each token ID -> its row in the word table
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # spots 0, 1, 2, 3 -> their rows
        x = tok_embeds + pos_embeds  # (4 tokens x 3)
        # Attention and the rest of GPT-2 will go here, in later steps.
        logits = self.out_head(x)  # (4 tokens x 50,257)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_TINY)
tokenizer = tiktoken.get_encoding("gpt2")

# One card of 4 tokens. The extra [ ] makes it a batch of 1 card, which is what the model expects.
in_idx = torch.tensor([tokenizer.encode("Your journey starts with")])
logits = model(in_idx)  # (1 card x 4 tokens x 50,257)
print("Output size:", tuple(logits.shape))

# For each spot, the guess = the token with the biggest number (argmax).
guesses = logits.argmax(dim=-1)[0]
for spot, guess in enumerate(guesses.tolist()):
    seen = tokenizer.decode(in_idx[0, : spot + 1].tolist())
    print(f"after {seen!r:<28} guess {tokenizer.decode([guess])!r}")
