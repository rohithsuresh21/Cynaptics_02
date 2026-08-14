import time
import sys
import torch
import config
from main import GPTModel
from utils import load_data, get_batch, estimate_loss

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

def train():
    train_data, val_data, tokenizer = load_data()
    log(f"Dataset loaded: {len(train_data) + len(val_data):,} chars | vocab size: {tokenizer.vocabulary_size}")
    log(f"Training on device: {config.DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = GPTModel(
        vocabulary_size=tokenizer.vocabulary_size,
        emb_dim=config.EMB_DIM,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        block_size=config.BLOCK_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")
    log(f"Training for {config.MAX_ITERS} iterations (eval every {config.EVAL_INTERVAL})...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    for iter in range(config.MAX_ITERS):
        step_start = time.time()

        if iter % config.EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            log(f"[EVAL] Step {iter:>5} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data)

        with torch.amp.autocast(device_type=device_type):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_time = time.time() - step_start
        elapsed = time.time() - start_time
        iters_left = config.MAX_ITERS - (iter + 1)
        eta = iters_left * step_time / 60
        log(f"Step {iter + 1:>5}/{config.MAX_ITERS} | loss: {loss.item():.4f} | {step_time:.2f}s/step | elapsed: {elapsed/60:.1f}m | ETA: {eta:.1f}m")

    torch.save({'model_state_dict': model.state_dict(), 'vocab': tokenizer.character}, config.CHECKPOINT_PATH)
    log(f"Training Complete. Model saved to {config.CHECKPOINT_PATH}.")

if __name__ == "__main__":
    train()