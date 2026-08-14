import torch

DATA_PATH = "shakespeare.txt"
TRAIN_SPLIT = 0.9  
BLOCK_SIZE = 256  
EMB_DIM = 384  
N_LAYER = 6  
N_HEAD = 6  
DROPOUT = 0.2  
TEMPERATURE = 1.0
BATCH_SIZE = 64  
LEARNING_RATE = 3e-4  
MAX_ITERS = 5000  
EVAL_INTERVAL = 100  
EVAL_ITERS = 200  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "model.pt"