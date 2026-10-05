import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text = """
science is the study of the natural world.
science uses observation experiments and evidence.
biology studies living organisms.
physics studies matter energy space and time.
chemistry studies substances and their transformations.
the earth moves around the sun.
water is important for life.
plants use sunlight to make food.
animals depend on plants and other organisms.
scientists collect data and build explanations.
"""

print(text)

# Creating word level tokenizer
words = text.lower().split()
print(words)
print("The original size:", len(words))

vocab = sorted(set(words))
print(vocab)
print("Vocabulary Size:", len(vocab))

# Now create mapping
word_to_id = {word: i for i , word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

print(id_to_word)

# Encode the text - Convert word into number
tokens = [word_to_id[word] for word in words]
print(tokens[:20])

# Creating training Sequence
context_size = 8
X = []
Y = []

for i in range(len(tokens) - context_size):
    x = tokens[i:i + context_size]
    y = tokens[i + 1:i + 1+ context_size]

    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # self-attention
        normalized_x = self.ln1(x)

        attention_output, _ = self.attention(
            normalized_x,
            normalized_x,
            normalized_x
        )

        x = x + attention_output
        # Feed forward Network
        x = x + self.feed_forward(self.ln2(x))

        return x

# Build the complete language Model
class MiniTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim=128,
            context_size=8,
            num_head=4,
            num_layers=2,
            dropout=0.1
        ):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Embedding
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

        # Transformer Block
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_dim,
                num_head,
                dropout
            )
            for _ in range(num_layers)
        ])

        # Final Normalization
        self.ln_final = nn.LayerNorm(embedding_dim)

        # Convert embedding to vocab scores
        self.output_layers = nn.Linear(embedding_dim, vocab_size)

        self.context_size = context_size


    def forward(self, x):
        batch_size, sequence_length = x.shape
        # Token Embedding
        token_embeddings = self.token_embedding(x)

        # Positions
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        position_embeddings = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        # Combine token + position info
        x = token_embeddings + position_embeddings

        # Transformer Block
        x = self.blocks(x)

        # Final Normalization
        x = self.ln_final(x)

        # Vocabulary Prediction
        logits = self.output_layers(x)

        return logits

model = MiniTransformer(
    vocab_size=len(vocab),
    embedding_dim=128,
    context_size=context_size,
    num_head=4,
    num_layers=2 
)

print(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.001
)

loss_function = nn.CrossEntropyLoss()

# training Loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X)

    loss = loss_function(
        logits.reshape(-1, len(vocab)),
        Y.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# generate Text
def generate(model, start_text, max_new_tokens=20):
    model.eval()
    words = start_text.lower().split()

    tokens = [
        word_to_id[word]
        for word in words
        if word in word_to_id
    ]

    if len(tokens) == 0:
        return ""

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        # Keep only the latest token
        x_context = x[:, -model.context_size:]

        # prediction
        logits = model(x_context)

        # take the final position
        logits = logits[:, -1, :]

        # convert scores into probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Select the next tokens
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Add it to sequence
        x = torch.cat([x, next_token], dim=1)

    generated_words = [
        id_to_word[token.item()]
        for token in x[0]
    ]

    return " ".join(generated_words)

print(generate(model, "science is", 20))
