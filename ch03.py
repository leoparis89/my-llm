# Chapter 3, first experiment: how similar are two words?
# Scores are picked by hand so the result is easy to check. A real model learns them.
scores = {
    "the": [0.1, 0.1, 0.1],
    "cat": [0.9, 0.2, 0.0],
    "dog": [0.9, 0.5, 0.0],
    "sat": [0.0, 0.0, 0.9],
}


# Similarity: multiply the matching scores and add them up. Bigger = more similar.
def similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


# Compare "cat" with every word
for word, word_scores in scores.items():
    print(f"cat vs {word}: {similarity(scores['cat'], word_scores):.2f}")
