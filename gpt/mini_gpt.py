# mini_gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        # GPT 核心：多层解码器（无 encoder-decoder attention）
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # 位置编码
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        
        # 创建 causal mask（关键！只允许看左边）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # 通过多层解码器
        for layer in self.layers:
            x = layer(x, x, tgt_mask=tgt_mask)  # self-attention only
        
        return self.lm_head(x)

# 训练示例（用你的 sentences 数据）
sentences = ["i love cats", "i love dogs", "you love animals"]
words = set(" ".join(sentences).split())
word2id = {w: i for i, w in enumerate(words)}
id2word = {i: w for w, i in word2id.items()}  # 反向映射：ID → 词
vocab_size = len(words)

# 构造训练数据：输入序列，目标是下一个词
def create_dataset(sentences, word2id):
    inputs, targets = [], []
    for s in sentences:
        tokens = [word2id[w] for w in s.split()]
        for i in range(len(tokens)-1):
            inputs.append(tokens[:i+1])      # 输入：前 i+1 个词
            targets.append(tokens[i+1])      # 目标：第 i+2 个词
    return inputs, targets

inputs, targets = create_dataset(sentences, word2id)

# 填充到相同长度（简单版）
max_len = max(len(x) for x in inputs)
padded_inputs = [x + [0]*(max_len-len(x)) for x in inputs]

model = MiniGPT(vocab_size, embed_dim=64, num_heads=2, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    input_tensor = torch.tensor(padded_inputs)
    target_tensor = torch.tensor(targets)
    
    logits = model(input_tensor)
    # 只计算最后一个位置的损失（因为前面位置的目标不确定）
    last_logits = logits[range(len(logits)), [len(x)-1 for x in inputs]]
    loss = criterion(last_logits, target_tensor)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 生成示例
def generate(model, start_tokens, max_new=5):
    model.eval()
    with torch.no_grad():
        tokens = start_tokens[:]
        for _ in range(max_new):
            input_tensor = torch.tensor([tokens])
            logits = model(input_tensor)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == word2id.get('eos', 0):  # 简单结束条件
                break
    return tokens

# 测试生成
start = [word2id['i']]
generated = generate(model, start)
print("生成结果:", " ".join(id2word[i] for i in generated))