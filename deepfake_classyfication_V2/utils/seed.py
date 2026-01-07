import os
import random
import torch
import numpy as np

"""Ustawiamy ziarno losowości dla różnych bibliotek, aby zapewnić powtarzalność wyników trenowania i ewaluacji modelu."""
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)