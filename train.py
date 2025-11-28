import os
import math
import torch
import pickle
import time
import argparse
from tokenizers import Tokenizer
from helper_functions import encode_corpus, load_wikipedia_text, make_dataloaders
from transformer_lm import TransformerLM
from tqdm import tqdm  


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

<<<<<<< HEAD
    parser.add_argument("--tokenizer_name", type=str, default="bytelevel")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no_tqdm", action="store_true")

=======
    parser.add_argument("--tokenizer_name", type=str, default="bpe", help="Name of the tokenizer to use, will search for tokenizers/ARGUMENT_tokenizer.json")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--stride", type=int, default=1, help="Stride for creating sequences in the dataloader")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars; useful for logging")
    parser.add_argument("--save_every", type=int, default=0, help="Save model every N epochs")
    parser.add_argument("--custom_name", type=str, default="", help="Custom name for saving model and history")
>>>>>>> fea3bb56f242487810d4d41ebdcb7127f6f48475
    parser.add_argument(
        "--target_chars",
        type=int,
        default=20 * 2_221_696,  # 20 x model params for vocab_size=16k
    )

    return parser.parse_args()



def train_model(tokenizer, train_loader, val_loader, model, vocab_size, device, learning_rate, num_epochs, disable_tqdm):
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "ppl": [], "throughput": []}

    num_batches = len(train_loader)
    batch_size = next(iter(train_loader))[0].shape[0]
    print("Num batches:", num_batches)
    print("Batch size:", batch_size)

    for epoch in range(num_epochs):
        start = time.time()
        model.train()

        train_loss_epoch = 0.0
        # Training loop with progress bar
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch", disable=disable_tqdm):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Evaluate on validation with progress bar
        model.eval()
        val_loss = 0.0
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", unit="batch", disable=disable_tqdm):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_ppl = math.exp(avg_val_loss)

        end = time.time()
        history["val_loss"].append(avg_val_loss)
        history["ppl"].append(val_ppl)
        history["throughput"].append(batch_size * num_batches / (end - start))

        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | ppl={val_ppl:.2f}")

        if SAVE_EVERY > 0 and epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)

    print("Training complete.")
    return model, history

if __name__ == "__main__":
    args = parse_args()

    TOKENIZER_NAME = args.tokenizer_name
    SEQ_LEN = args.seq_len
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    TARGET_CHARS = args.target_chars
    STRIDE = args.stride
    TQDM_DISABLED = args.no_tqdm
    SAVE_EVERY = args.save_every
    CUSTOM_NAME = args.custom_name
    TOKENIZER_PATH = f"tokenizers/{TOKENIZER_NAME}_tokenizer.json"
    if CUSTOM_NAME:
        HISTORY_PATH = f"history/{CUSTOM_NAME}_training_history.pkl"
        MODEL_PATH = f"models/{CUSTOM_NAME}_transformer.pth"
    else:   
        HISTORY_PATH = f"history/{TOKENIZER_NAME}_training_history.pkl"
        MODEL_PATH = f"models/{TOKENIZER_NAME}_transformer.pth"
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    model = TransformerLM(vocab_size=vocab_size, max_seq_length=SEQ_LEN).to(device)
    model = torch.nn.DataParallel(model)
    
    text_en = load_wikipedia_text(language="en", target_chars=TARGET_CHARS // 2)
    text_ru = load_wikipedia_text(language="ru", target_chars=TARGET_CHARS // 2)
    text = text_en + text_ru

    ids = encode_corpus(tokenizer, text)
    print(f"Total tokens collected: {len(text):,}")

    # train_loader, val_loader, test_loader = make_dataloaders(ids[:SEQ_LEN*2**8], seq_len=SEQ_LEN, batch_size=BATCH_SIZE, stride=STRIDE)
    train_loader, val_loader, test_loader = make_dataloaders(ids, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, stride=STRIDE)

    trained_model, training_history = train_model(
        tokenizer,
        train_loader,
        val_loader,
        model,
        vocab_size,
        device,
        LEARNING_RATE,
        NUM_EPOCHS,
        TQDM_DISABLED
    )

    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(training_history, f)
