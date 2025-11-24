import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "i love this movie",
    "this is amazing",
    "i really like this film",
    "the story is great",

    "i hate this movie",
    "this is bad",
    "i really dislike this film",
    "the story is terrible",
]

labels = [
    1, 1, 1, 1,   
    0, 0, 0, 0, 
]

words = set(" ".join(sentences).split())
word2id = {w : i+1 for i, w in enumerate(words)}
word2id["<PAD>"] = 0
vocab_size = len(word2id)

def encode_sentence(sentence, max_len = 6):
    ids = [word2id[w] for w in sentence.split()]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

encoded_sentences = [encode_sentence(s) for s in sentences]

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb)
        last_hidden = hidden.squeeze(0)
        logits = self.fc(last_hidden)
        return logits

embed_dim = 8
hidden_dim = 16
num_classes = 2
batch_size = len(encoded_sentences)

model = SimpleRNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

inputs = torch.tensor(encoded_sentences)
targets = torch.tensor(labels)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
test_sentence = "i love"
test_input = torch.tensor([encode_sentence(test_sentence)])
logits = model(test_input)
pred = logits.argmax(dim=1).item()
print(f"句子：'{test_sentence}'，预测情感为：{'正面' if pred == 1 else '负面'}")
