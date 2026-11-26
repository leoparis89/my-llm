import tiktoken
import torch
import torch.nn as nn
import math

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

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # token_ids = one card: the IDs of up to 4 words. (The book calls it in_idx.)
    #   [7120, 7002, 4940, 351]   "Your journey starts with"
    # (The book feeds many cards at once, as a grid. We'll do that later, when training on many cards.)
    def forward(self, token_ids):
        card_len = len(token_ids)

        tok_embeds = self.tok_emb(token_ids)

        pos_embeds = self.pos_emb(torch.arange(card_len, device=token_ids.device))

        x = tok_embeds + pos_embeds  # (4 words x 3)

        logits = self.out_head(x)
        return logits

def generate_text(model, starting_token_ids, context_size, length_of_dance):
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
token_ids = torch.tensor(tokenizer.encode("Your journey starts with a really long sentence"))

# The whole story "The Verdict" (downloaded by ch02.py) -> one long list of token IDs.
#   "I HAD always thought Jack Gisburn rather a cheap genius..." -> [40, 367, 2885, 1464, 1807, 3619, ...]
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()  # the whole story as one string: 20,479 characters
all_ids = torch.tensor(tokenizer.encode(raw_text))  # 5,145 token IDs

def calc_loss(model, *, input_card, right_answers_card):
    logits = model(input_card)  # scores: 4 words x 50,257
    # Compare the 4 guesses' scores with the 4 right answers.
    loss = nn.CrossEntropyLoss()(logits, right_answers_card)
    return loss

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def train_step(model, input_card, right_answers_card):
    loss = calc_loss(model, input_card=input_card, right_answers_card=right_answers_card)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

amount_of_cards = len(all_ids) - GPT_CONFIG_TINY["context_length"] 


for _i in range(2):
    for i in range(amount_of_cards):
        input_card = all_ids[i:i+GPT_CONFIG_TINY["context_length"]]
        right_answers_card = all_ids[i+1:i+GPT_CONFIG_TINY["context_length"]+1]
        loss = train_step(model, input_card, right_answers_card)


starting_input_card = token_ids[:GPT_CONFIG_TINY["context_length"]]
generated_text = generate_text(model, starting_input_card, GPT_CONFIG_TINY["context_length"], 29)

print(f"generated_text: {tokenizer.decode(generated_text.tolist())}")
print(f"*** end of text ***")
