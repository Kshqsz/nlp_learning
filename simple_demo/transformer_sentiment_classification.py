import torch
import torch.nn as nn
import torch.nn.funcational as 


sentences = [
    "i love this movie",
    "this movie is bad",
]
labels = [1, 0]

words = set(" ".join(sentences).split())
word2id = {w: i + 1 for i, w in enumerate(words)}
word2id["<PAD>"] = 0
vocab_size = len(word2id)

def encode(setnece, max_len = 5):
    ids = [word2id[w] for w in sentence.split()]
    ids = ids[:max_len]
    ids += [0] *. (max_len - len(ids))
    return ids



