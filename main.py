import torch
import torch.nn as nn
from torch.nn import functional as F

class CharTokenizer:
    def __init__(self, text):
        self.character = sorted(list(set(text)))
        self.vocabulary_size = len(self.character)
        self.character_to_index = {ch: i for i, ch in enumerate(self.character)}
        self.index_to_character = {i: ch for i, ch in enumerate(self.character)}

    def encode(self, text):
        return [self.character_to_index[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.index_to_character[idx] for idx in indices])
        
class TokenEmbeddings(nn.Module):
    def __init__(self, vocabulary_size, emd_dim, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, emd_dim)
        self.position_embedding_table = nn.Embedding(block_size, emd_dim)

    def forward(self, index):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))
        return tok_emb + pos_emb

class SelfAttention(nn.Module):
    def __init__(self, block_size, emd_dim, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emd_dim, head_size, bias=False)
        self.query = nn.Linear(emd_dim, head_size, bias=False)
        self.value = nn.Linear(emd_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, emd_dim, head_size, n_head, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(block_size, emd_dim, head_size, dropout) for _ in range(n_head)])
        self.projection = nn.Linear(emd_dim, emd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class MultilayerPerceptron(nn.Module):
    def __init__(self, emd_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emd_dim, 4 * emd_dim),
            nn.ReLU(),
            nn.Linear(4 * emd_dim, emd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, block_size, emd_dim, n_head, dropout=0.1):
        super().__init__()
        head_size = emd_dim // n_head
        self.attention = MultiHeadAttention(block_size, emd_dim, head_size, n_head, dropout)
        self.mlp = MultilayerPerceptron(emd_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(emd_dim)
        self.layer_norm2 = nn.LayerNorm(emd_dim)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocabulary_size, emb_dim, n_layer, n_head, block_size, dropout=0.1):
        super().__init__()
        self.embeddings = TokenEmbeddings(vocabulary_size, emb_dim, block_size)
        self.blocks = nn.Sequential(*[TransformerBlock(block_size, emb_dim, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embeddings(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration --- #
# Paths relative to the script location
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakespeare.txt")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

# --- API Routes --- #

@app.route('/status', methods=['GET'])
def get_status():
    # In a real application, you'd return actual system metrics here.
    return jsonify({
        "status": "Operational",
        "cpu_load": "15%",
        "memory_usage": "5.1GB",
        "active_processes": 4,
        "last_activity": "2026-08-14 14:30:00"
    })

@app.route('/generate', methods=['POST'])
def generate_output():
    data = request.json
    prompt = data.get('prompt', '')
    length = data.get('length', 200)

    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        return jsonify({
            "success": False,
            "message": "Model not trained. Please initiate training first.",
            "log": "Error: model.pt not found. Call /train to train the model before generating."
        })

    try:
        result = subprocess.run(
            ["python", "generate.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=300
        )
        output = result.stdout + result.stderr
        if not output.strip():
            output = "Generation executed (no output captured)."
        return jsonify({
            "success": True,
            "output": output,
            "log": "Generation initiated."
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Generation timed out.",
            "log": "Generation process timed out after 5 minutes."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "log": str(e)
        })

@app.route('/train', methods=['POST'])
def initiate_train():
    data = request.json
    try:
        # Check if data file exists, if not, train.py will attempt to download it
        # We run train.py. It may take some time (config.MAX_ITERS = 5000).
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=600 # 10 minutes timeout for training
        )
        output = result.stdout + result.stderr
        if not output.strip():
            output = "Training executed (no output captured.)"
        
        # --- Parse loss values from train.py output ---
        latest_loss_line = ""
        for line in output.split('\n'):
            if 'Train loss' in line or 'Val loss' in line:
                latest_loss_line = line
        
        # Create a summary message highlighting the latest loss
        if latest_loss_line:
            summary = f"Training completed. {latest_loss_line}"
        else:
            summary = output # Fallback to full output if pattern not found
        # ----------------------------------------------
        
        # Check if model.pt was created successfully
        if os.path.exists(MODEL_FILE):
            return jsonify({
                "success": True,
                "message": "Training completed. Model saved.",
                "log": summary
            })
        else:
            # This case happens if training failed silently or didn't save
            return jsonify({
                "success": False,
                "message": "Training did not complete or save model.",
                "log": summary + "\n(Note: model.pt was not created.)"
            })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Training timed out.",
            "log": "Training process timed out after 10 minutes. This is normal for CPU training on large iters."
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)