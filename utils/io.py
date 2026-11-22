#path: full_code/utils/io.py
import os
import random
import numpy as np
import torch
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
