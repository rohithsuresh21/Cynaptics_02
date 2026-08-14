import torch
import torch.nn.functional as F
import config
from main import GPTModel, CharTokenizer

def text_generate(model, tokenizer, seed="", max_new_tokens=500):
    model.eval()
    idx = torch.tensor(tokenizer.encode(seed), dtype=torch.long).unsqueeze(0).to(config.DEVICE) if seed else torch.zeros((1, 1), dtype=torch.long).to(config.DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / config.TEMPERATURE
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    with open('shakespeare.txt', 'r', encoding='utf-8') as f:
     text = f.read()
    tokenizer = CharTokenizer(text)
    checkpoint = torch.load('model.pt', map_location=config.DEVICE, weights_only=True)
    tokenizer = CharTokenizer("".join(checkpoint['vocab']))
    model = GPTModel(tokenizer.vocabulary_size, config.EMB_DIM, config.N_LAYER, config.N_HEAD, config.BLOCK_SIZE).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(text_generate(model, tokenizer, seed="JULIET:", max_new_tokens=1000))