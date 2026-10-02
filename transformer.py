import torch 
import torch.nn as nn
import torch.nn.functional as F

# Small text dataset
text = "Hello, This is Sachin Bista also known as a great programmer. he has knowledge in the machine learning and artificial intelligence. I know the fundamental concept of building transformer model."  * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# Small Settings
batch_size = 16
block_size = 8
embedding_dim = 16
learning_rate = 0.01
steps = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The device is {device}")

# Create Training Example
def get_batch():
    starts = torch.randint(0, len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])

    return x, y

# Tiny Transformer
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # each character ID became learned vector
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Transformer need position information explicitly
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # Turn each embedding into Query, Key, Value vectors
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Convert final hidden vocabulary into vocabulary scores
        self.output = nn.Linear(embedding_dim, vocab_size)

        # Causal mask: lower traingular is 1, upper traingular is 0
        # This stop character looking into the future
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

def forward(self, x, target=None):
    # X shape : (B,T)
    # B = batch size, T = sequence length
    B, T = x.shape

    # token embedding shape
    token_vectors = self.token_embedding(x)

    # Positions: [0,1,2,3...., T-1]
    positions = torch.arange(T, device=x.device)

    # position vector shape
    position_vectors = self.position_embedding(positions)

    # Add token identiry and token position
    h = token_vectors + position_vectors

    # Self - Attention
    q = self.query(h)
    k = self.query(h)
    v = self.value(h)

    attention_scores = q * k.transpose(-2, -1)
    attention_scores = attention_scores / (embedding_dim ** 0.5)

    # Do not allow future position
    attention_scores = attention_scores.masked_fill(
        self.mask[:T, :T] == 0,
        float("-inf")
    )

    # turn scores into probablities
    attention_weights = F.softmax(attention_scores, dim=-1)

    h = attention_weights @ v

    # predict the next character
    logits = self.output(h)

    loss = None
    if target is not None:
        loss = F.cross_entropy(
            logits.reshape(B * T, vocab_size),
            target.reshape(B*T)
        )

    return logits, loss
@torch.no_grad()
def generate(self, start_text, new_characters=100):
    self.eval()

    ids = torch.tensor([encode(start_text)], dtype=torch.long, device=device)

    for _ in range(new_characters):
        # Model only accepts at most block_size characters.
        context = ids[:, -block_size:]

        logits, _ = self(context)

        # Last position predicts the next character.
        next_logits = logits[:, -1, :]
        # Convert scores into probabilities.
        probabilities = F.softmax(next_logits, dim=-1)

        # Randomly sample a character.
        next_id = torch.multinomial(probabilities, num_samples=1)

        # Add it to the sequence.
        ids = torch.cat([ids, next_id], dim=1)

        return decode(ids[0].tolist())

model = TinyTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step in range(steps):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

print(model.generate("hello ", new_characters=120))