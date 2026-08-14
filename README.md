# Character-Level GPT Custom Dataset Transformer

> A '**decoder-only Transformer architecture**' that learns and mimics the linguistic style of any provided text file, trained character by character.

---

## Project Structure

Each module has a distinct responsibility, keeping the codebase clean and modular.

| File | Nickname | Responsibility |
|------|----------|----------------|
| `config.py` |  **The Blueprint** | Houses all hyperparameters — `BATCH_SIZE`, `BLOCK_SIZE`, `N_LAYER`, `LEARNING_RATE`. Any change here reshapes the entire model. |
| `dataloader.py` |  **The Delivery Truck** | Reads `input.txt`, encodes it into integers, and chops it into batches so the GPU isn't overwhelmed. |
| `train.py` |  **The Gym** | Runs the training loop — feeds data to the model, calculates loss, and updates weights via AdamW. |
| `generate.py` |  **The Stage** | Loads saved weights, accepts a seed like `"JULIET:"`, and predicts characters one by one up to `max_new_tokens`. |
| `main.py` |  **The Manager** | The entry point — decides whether to launch a fresh training session or run a quick generation test. |
| `utils.py` |  **The Toolbox** | Holds the `CharTokenizer` class plus helpers for estimating loss and saving checkpoints. loads the data and batch generarion is implemnted |

---

##  How It Works — The Pipeline

### 1.  Tokenization

Computers can't read `"A"` or `"B"` directly. The `CharTokenizer` in `utils.py` bridges that gap by building a character-level lookup table.

- **Encoding:** Converts `"JULIET"` → `[20, 31, 22, 19, 15, 30]`
- **Decoding:** Reverses the process to reconstruct readable text from integers

---

### 2.  Embeddings

A raw integer like `20` carries no semantic meaning about the letter `"J"`. Embeddings fix this.

- **Token Embeddings** — Each integer is mapped to a dense vector (e.g., 128 or 256 floats), giving the model a rich representation of each character.
- **Positional Embeddings** — Since Transformers process all characters in a block simultaneously, they lose track of order. A positional vector is added to signal that `"J"` came first and `"U"` came second.

---

### 3.  Self-Attention

The core of the Transformer. Every character dynamically decides which other characters in the sequence matter most.

> **Queries, Keys & Values:** Each character asks *"Which other characters are relevant to me right now?"*

**Example:** In `"Juliet"`, the final `"t"` attends strongly to `"e"` and `"i"` — contextual clues that it's part of a proper name, not a standalone word.

---

### 4.  Feed-Forward Network (FFN)

After self-attention has gathered context from across the sequence, the data flows through a position-wise Feed-Forward Network.

This is where the model **processes and encodes patterns** — translating raw contextual signals into meaningful representations for prediction.

---

### 5.  Softmax Output

At the final layer, the model produces a **score for every character in its vocabulary**. The Softmax function converts these raw logits into a probability distribution, from which the next character is sampled.

---

##  Key Challenges

### 1.  GPU Engagement (CUDA Setup)
Getting Python 3.13, PyTorch, and CUDA 12.x to cooperate — and ensuring the GPU actually *engages* instead of silently defaulting to slow CPU execution — is the single biggest environment hurdle.

### 2.  Hyperparameter Balance
`config.py` must be carefully tuned:
- **Learning rate too high** → Loss spikes to `~10.0` and stalls
- **Learning rate too low** → Loss barely moves from `~4.0` after hours of training
- A well-calibrated config is the difference between convergence and chaos.

### 3.  Dataset Scaling
The Tiny Shakespeare dataset is a great starting point, but the model will eventually **memorize it entirely**. Genuine generalization requires a dataset roughly **100× larger** — meaning real effort around sourcing, cleaning, and formatting raw text.

---

##  End-to-End Flow

```
input.txt
    │
    ▼
CharTokenizer (utils.py)     ← Encode characters → integers
    │
    ▼
DataLoader (dataloader.py)   ← Chunk into batches
    │
    ▼
Transformer Model            ← Token Embed → Pos Embed → Self-Attention → FFN → Softmax
    │
    ▼
Training Loop (train.py)     ← Forward pass → Loss → Backprop → AdamW update
    │
    ▼
Saved Checkpoint (utils.py)
    │
    ▼
generate.py                  ← Seed → Predict next char → Repeat → Output text
```

---

