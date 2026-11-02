import tiktoken
import torch

# The final machine, built one piece at a time.
# It follows the book's GPT-2 model (chapter 4), so at the end
# OpenAI's trained GPT-2 numbers can be loaded into it.

# GPT-2 small sizes (the book's GPT_CONFIG_124M)
vocab_size = 50257  # tokens in GPT-2's list
context_length = 256  # most tokens it reads at once (GPT-2 allows 1024; the book trains with 256)
emb_dim = 768  # numbers per token (we used 3 in ch02.py)


# A torch.nn.Module is PyTorch's box for a model:
#   __init__ creates the tables and filters (all random at first, adjusted by training)
#   forward  is what happens when you call model(token_ids)
class GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_table = torch.nn.Embedding(vocab_size, emb_dim)  # ch02 step 3; the book calls it tok_emb
        self.position_table = torch.nn.Embedding(context_length, emb_dim)  # ch02 step 4; the book calls it pos_emb
        # A filter that turns each token's 768 numbers into 50,257 numbers:
        # one per token in the list, saying how strongly the model thinks it comes next.
        # The book calls the filter out_head and its output "logits".
        self.next_token_filter = torch.nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        spots = torch.arange(token_ids.shape[-1])  # [0, 1, 2, ...] one per token on the card
        x = self.word_table(token_ids) + self.position_table(spots)
        # Attention and the rest of GPT-2 will go here, in later steps.
        return self.next_token_filter(x)


torch.manual_seed(123)
model = GPT()
tokenizer = tiktoken.get_encoding("gpt2")

# One card of 4 tokens. The extra [ ] makes it a batch of 1 card, which is what the model expects.
token_ids = torch.tensor([tokenizer.encode("Your journey starts with")])
next_token_numbers = model(token_ids)  # (1 card x 4 tokens x 50,257)
print("Output size:", tuple(next_token_numbers.shape))

# For each spot, the guess = the token with the biggest number (argmax).
guesses = next_token_numbers.argmax(dim=-1)[0]
for spot, guess in enumerate(guesses.tolist()):
    seen = tokenizer.decode(token_ids[0, : spot + 1].tolist())
    print(f"after {seen!r:<28} guess {tokenizer.decode([guess])!r}")
