# trial.py

import torch
from loss import get_brats_loss

def main():
    B, C, D, H, W = 2, 5, 16, 32, 32
    logits = torch.randn(B, C, D, H, W)
    # Create random one-hot targets
    idx = torch.randint(0, C, (B, D, H, W))
    target = torch.nn.functional.one_hot(idx, num_classes=C).permute(0,4,1,2,3).float()

    loss_fn = get_brats_loss()
    losses = loss_fn(logits, target)
    print(losses)

if __name__=="__main__":
    main()



