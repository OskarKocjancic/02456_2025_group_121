import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Dataset class
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len=20):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx + self.seq_len]
        y = self.token_ids[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# --- LM model
class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = 4,
            batch_first = True
        )

        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor):
        ws = self.embeddings(token_ids)
        # shift right by one
        w0 = torch.zeros((ws.size(0), 1, self.embedding_dim), device=ws.device, dtype=ws.dtype)
        ws_shifted = torch.cat([w0, ws[:, :-1, :]], dim=1)
        hidden_states, _ = self.rnn(ws_shifted)
        logits = self.proj(hidden_states)
        return logits

    def sample(self, batch_size=1, num_steps=20, temperature: float = 1.0):
        device = self.embeddings.weight.device
        token_ids = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
        for t in range(num_steps):
            logits = self.forward(token_ids)
            logits_t = logits[:, -1:, :] / temperature
            p = torch.distributions.Categorical(logits=logits_t)
            next_tokens = p.sample()
            token_ids = torch.cat([token_ids, next_tokens], dim=1)
        return token_ids

# --- Training / evaluation
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# --- Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_articles = 1000
    # 1) Load Wikimedia dataset
    # Example: English Wikipedia dump 20231101
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{max_articles}]")  # lang=en
    # Each example has e.g. ‘text’ field (the article content) :contentReference[oaicite:4]{index=4}

    # 2) Choose tokenizer
    # tokenizer_name = "bert-base-uncased"
    tokenizer_name = "gpt2"
    
    # THIS IS PRETRAINED, WE NEED TO DO IT OURSELVES !
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 3) Tokenize dataset (we’ll use a subset for speed)
    texts = dataset["text"]
    token_ids = []
    for t in texts:
        # ids = tokenizer.encode(t, add_special_tokens=True)
        ids = tokenizer.encode(t, add_special_tokens=True, max_length=512, truncation=True)
        token_ids.extend(ids)

    # 4) Split into train/test
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[: split_idx]
    test_ids = token_ids[split_idx :]

    # 5) Build dataloaders
    seq_len = 10
    train_ds = TextDataset(train_ids, seq_len=seq_len)
    test_ds = TextDataset(test_ids, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # 6) Set up model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    hidden_dim = 256
    model = RNNLM(vocab_size, embedding_dim, hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 7) Train & evaluate
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", end=" —`")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # 8) Sample and decode
    sample_ids = model.sample(batch_size=2, num_steps=30, temperature=1.0)
    print("Sampled text:", [tokenizer.decode(ids.tolist()) for ids in sample_ids])
