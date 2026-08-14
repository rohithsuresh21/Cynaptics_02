import torch
import config
import os
import requests
from main import CharTokenizer

def load_data():
    if not os.path.exists(config.DATA_PATH):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        with open(config.DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(text)
    with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
        data = f.read()
    tokenizer = CharTokenizer(data)
    tokenized_data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    n = int(config.TRAIN_SPLIT * len(tokenized_data))
    return tokenized_data[:n], tokenized_data[n:], tokenizer

def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i+config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
    return x.to(config.DEVICE), y.to(config.DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out