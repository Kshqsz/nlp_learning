import torch
import torch.nn as nn
import torch.nn.functional as F


sentences = [
    "i love this movie",
    "this movie is bad",
]
labels = [1, 0]

words = set(" ".join(sentences).split())
word2id = {w: i + 1 for i, w in enumerate(words)}
word2id["<PAD>"] = 0
vocab_size = len(word2id)

def encode(sentence, max_len = 5):
    ids = [word2id[w] for w in sentence.split()]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

inputs = torch.tensor([encode(s) for s in sentences])
targets = torch.tensor(labels)


class MiniSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (x.size(-1) ** 0.5)
        attn = F.softmax(scores, dim = -1)
        out = attn @ V
        return out, attn

class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, 5, dim))

        self.attn = MiniSelfAttention(dim)
        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x) + self.pos
        
        attn_out, attn_weights = self.attn(emb)
        x = self.norm1(emb + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        cls_token = x[:, 0]
        logits = self.classifier(cls_token)
        return logits, attn_weights

model = MiniTransformerEncoder(vocab_size, dim = 8, num_classes = 2)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(50):
    opt.zero_grad()
    logits, attn = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

test = torch.tensor([encode("this movie is bad")])
logits, attn = model(test)
pred = logits.argmax(dim = 1).item()

print("预测：", "正面" if pred == 1 else "负面")
print("注意力矩阵：\n", attn[0].detach())
