import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = [
    "i love this movie",
    "this movie is bad",
]

labels = [1, 0]

words = set(" ".join(sentences).split())
word2id = {w : i + 1 for i, w in enumerate(words)}
word2id["<PAD>"] = 0
vocab_size = len(word2id)

def encode(sentence, max_len = 5):
    ids = [word2id[w] for w in sentence.split()]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

inputs = torch.tensor([encode(s) for s in sentences])
targets = torch.tensor(labels)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, 5, dim))

        layer = nn.TransformerEncoderLayer(
            d_model = dim,
            nhead = num_heads,
            dim_feedforward = dim * 4,
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers = num_layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x) + self.pos
        enc = self.encoder(emb)

        cls_token = enc[:, 0]
        logits = self.classifier(cls_token)
        return logits

model = TransformerClassifier(
    vocab_size = vocab_size,
    dim = 16,
    num_heads = 2,
    num_layers = 2,
    num_classes = 2
)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(50):
    opt.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    opt.step() 

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

test = torch.tensor([encode("i love this movie")])
logits = model(test)
pred = logits.argmax(dim=1).item()
print("预测：", "正面" if pred == 1 else "负面")