from datasets import load_dataset
from typing import List, Iterable
import torch
from torch.utils.data import Dataset, DataLoader
import os


def load_wikipedia_text(language, target_chars, cache_dir=None, split="train[:70%]") :
    # check if /dtu/blackhole by accessing environment variable

    if cache_dir is None:
        blackhole = os.environ.get('BLACKHOLE')
        if blackhole:
            cache_dir = os.path.realpath(blackhole)
    



    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{language}",
        split=split,
        # trust_remote_code=True,
        # uncomment this line to save to $BLACKHOLE in HPC
        cache_dir=cache_dir,
    )

    texts = []
    chars_collected = 0
    dataset = dataset.shuffle(seed=42)

    for row in dataset:
        article = row["text"]
        article_len = len(article)

        if chars_collected + article_len <= target_chars:
            texts.append(article)
            chars_collected += article_len
        else:
            # Only take the portion we need to reach the target
            remaining = target_chars - chars_collected
            texts.append(article[:remaining])
            chars_collected += remaining
            break

    # print(f"{language}: collected {chars_collected:,} characters from {len(texts):,} articles")
    return texts


## converts long list of token ids into many small overlapping training samples for next-token language model
## produces sliding windows of seq_len size
## input x are token sequences
## outputs y are the same sequences shifted by one position - meaning predict the next token
## this way, the model learns - given the sequence, what comes next?
class NextTokenDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int, stride: int):
        self.ids = token_ids
        self.seq_len = seq_len

        # Number of training examples (each is a sliding window of length seq_len)
        # self.n = max(0, len(self.ids) - self.seq_len)
        self.stride = stride

        max_start = len(self.ids) - seq_len - 1  # -1 because y starts at start+1
        if max_start <= 0:
            self.starts = []
        else:
            self.starts = list(range(0, max_start + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + 1 + self.seq_len]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def encode_corpus(tokenizer, texts: Iterable[str], print_out=True) -> List[int]:
    """Concatenate all article token-ids into one long stream."""
    all_ids: List[int] = []

    # Define a valid separator token ID
    sep_id = tokenizer.token_to_id("[EOS]")

    for t in texts:
        try:
            enc = tokenizer.encode(t)
            ids = enc.ids if hasattr(enc, "ids") else enc  # handle different API types
            all_ids.extend(ids)
            all_ids.append(sep_id)  # separator for stability
        except Exception as e:
            # If something fails (bad text, unknown chars), skip that article
            print(f"Skipped an article due to encoding error: {e}")
            continue

    # Final safety check — all IDs must be < vocab_size
    vocab_size = tokenizer.get_vocab_size()
    before_count = len(all_ids)
    all_ids = [i for i in all_ids if isinstance(i, int) and 0 <= i < vocab_size]
    if print_out:
        if len(all_ids) < before_count:
            print(f"Removed {before_count - len(all_ids)} invalid token IDs from corpus")

        print(f"Encoded {len(texts)} texts into {len(all_ids)} token IDs")
        print(f"Vocabulary size: {vocab_size}")
    return all_ids


# use 70% train, 10% val, 20% test split
def make_dataloaders(token_ids: List[int], seq_len=256, batch_size=32, stride=1):
    n = len(token_ids)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.1)

    train_ids = token_ids[:train_end]
    val_ids = token_ids[train_end:val_end]
    test_ids = token_ids[val_end:]

    train_ds = NextTokenDataset(train_ids, seq_len=seq_len, stride=stride)
    val_ds = NextTokenDataset(val_ids, seq_len=seq_len, stride=stride)
    test_ds = NextTokenDataset(test_ids, seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader
