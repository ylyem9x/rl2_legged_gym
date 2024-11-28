from typing import Any
import torch

class uniform_noise:
    def __init__(self, min = 0.9, max = 1.1, mode = "scale"):
        self.min = min
        self.max = max
        self.mode = mode
        if mode == "scale":
            if min < 0:
                raise ValueError("noise will change the direction!")
        elif mode == "add":
            if min > 0:
                print("[WARNING] Adding unsymmetrical noise!")
        else:
            raise ValueError(f"Unsupported mode value, expected scale/add, but receive {mode}")

    def __call__(self, tensor:torch.Tensor) -> torch.Tensor:
        if self.mode == "scale":
            return tensor * (torch.rand_like(tensor) * (self.max - self.min) + self.min)
        else:
            return tensor + torch.rand_like(tensor) * (self.max - self.min) + self.min

    def __str__(self):
        s = f"mode:{self.mode}, min:{self.min}, max:{self.max}"
        return s