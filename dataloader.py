import torch
import requests
import os
from typing import Tuple

class ShakespeareDataLoader:
    def __init__(self, data_path="shakespeare.txt", train_split=0.9, block_size=256, batch_size=64, device='cpu'):
        self.data_path = data_path
        self.train_split = train_split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        self.prepare_data()

    def download_dataset(self):
        if not os.path.exists(self.data_path):
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url).text
            with open(self.data_path, 'w', encoding='utf-8') as f:
                f.write(text)

    def prepare_data(self):
        self.download_dataset()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(self.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]

    def decode(self, l: list) -> str:
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split='train') -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)