import torch

# Chapter 3, first experiment: how similar are two words?
# Scores are picked by hand so the result is easy to check. A real model learns them.
scores = {
    "the": torch.tensor([0.1, 0.1, 0.1]),
    "cat": torch.tensor([0.9, 0.2, 0.0]),
    "dog": torch.tensor([0.9, 0.5, 0.0]),
    "sat": torch.tensor([0.0, 0.0, 0.9]),
}

# Similarity = dot product. Bigger = more similar.
for word, word_scores in scores.items():
    print(f"cat vs {word}: {torch.dot(scores['cat'], word_scores):.2f}")
