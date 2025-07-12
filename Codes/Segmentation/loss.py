import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import TverskyLoss, HausdorffDTLoss
from monai.utils import LossReduction
import matplotlib.pyplot as plt
import numpy as np


class VolumeAwareTverskyLoss(nn.Module):
    def __init__(
        self,
        include_background=False,
        softmax=True,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        class_weights=None,
        baseline_volumes=None,
        epsilon=1e-5,
        reduction="mean",
        loss_weights=(1.0, 0.0),  # (tversky, hd)
        max_weight_factor=2.5,
        hd_percentile=95.0,
    ):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.epsilon = epsilon
        self.reduction = reduction
        self.loss_weights = loss_weights
        self.max_weight_factor = max_weight_factor

        cw = class_weights or [1.0, 2.0, 1.5, 2.5, 1.5]
        bv = baseline_volumes or [0.0, 2000.0, 8000.0, 1000.0, 3000.0]
        self.n_classes = len(cw)

        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            alpha=tversky_alpha,
            beta=tversky_beta,
            reduction=LossReduction.MEAN,
        )

        self.hd_loss = HausdorffDTLoss(
            include_background=include_background,
            reduction=LossReduction.NONE,
        )

        self.register_buffer("static_w", torch.tensor(cw, dtype=torch.float32))
        self.register_buffer("baseline_w", torch.tensor(bv, dtype=torch.float32))

    def _compute_dynamic_weights(self, onehot):
        vols = onehot.sum(dim=(2, 3, 4))
        baseline = self.baseline_w.to(vols.device)
        static = self.static_w.to(vols.device)
        with torch.no_grad():
            mult = torch.sqrt(baseline / torch.clamp(vols, min=self.epsilon))
            mult = torch.clamp(mult, max=self.max_weight_factor)
            eff = static * mult
            dyn = eff / torch.clamp(eff.max(dim=1, keepdim=True)[0], min=self.epsilon)
        return dyn

    def forward(self, pred, target):
        assert pred.dim() == 5 and target.dim() == 5

        dynamic_w = self._compute_dynamic_weights(target)
        self.tversky.class_weight = dynamic_w.clone()

        tversky_loss = self.tversky(pred, target)
        probs = F.softmax(pred, dim=1) if self.softmax else pred
        hd_pc = self.hd_loss(probs, target)
        if not self.include_background:
            hd_pc = hd_pc[:, 1:]
        hd_loss = hd_pc.mean(dim=1)

        w1, w2 = self.loss_weights
        total = w1 * tversky_loss + w2 * hd_loss

        with torch.no_grad():
            pred_soft = F.softmax(pred, dim=1) if self.softmax else pred
            pred_bin = (pred_soft > 0.5).float()
            tp = (pred_bin * target).sum(dim=(0, 2, 3, 4))
            fp = (pred_bin * (1 - target)).sum(dim=(0, 2, 3, 4))
            fn = ((1 - pred_bin) * target).sum(dim=(0, 2, 3, 4))

        return {
            "loss": total.mean(),
            "tversky_loss": tversky_loss.mean(),
            "hd_loss": hd_loss.mean(),
            "dynamic_weights": dynamic_w,
            "per_sample_loss": total,
            "per_class_tp": tp,
            "per_class_fp": fp,
            "per_class_fn": fn,
        }

def compute_metrics(tp, fp, fn, epsilon=1e-8):
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tp / (tp + fp + epsilon)
    return {
        "dice": dice,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

# === Demo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, C, D, H, W = 2, 5, 64, 64, 64
pred_logits = torch.randn(B, C, D, H, W, device=device)
target_seg = torch.randint(0, C, (B, D, H, W), device=device)
target_onehot = torch.nn.functional.one_hot(target_seg, num_classes=C).permute(0, 4, 1, 2, 3).float()

loss_fn = VolumeAwareTverskyLoss(
    include_background=False,
    softmax=True,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    loss_weights=(1.0, 0.0),
).to(device)

loss_output = loss_fn(pred_logits, target_onehot)
print("Total Loss:", loss_output["loss"].item())
print("Tversky Loss:", loss_output["tversky_loss"].item())
print("Per-class TP:", loss_output["per_class_tp"].cpu().numpy())
print("Per-class FP:", loss_output["per_class_fp"].cpu().numpy())
print("Per-class FN:", loss_output["per_class_fn"].cpu().numpy())

metrics = compute_metrics(
    loss_output["per_class_tp"],
    loss_output["per_class_fp"],
    loss_output["per_class_fn"]
)
print("\n--- Per-Class Metrics ---")
for i, (d, s, p) in enumerate(zip(metrics['dice'], metrics['sensitivity'], metrics['specificity'])):
    print(f"Class {i}: Dice={d:.4f}, Sensitivity={s:.4f}, Specificity={p:.4f}")

with torch.no_grad():
    pred_mask = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
    fig, axes = plt.subplots(2, C-1, figsize=(15, 5))
    slice_idx = D // 2
    for c in range(1, C):
        axes[0, c-1].imshow(pred_mask[0, slice_idx].cpu() == c, cmap='Reds')
        axes[0, c-1].set_title(f"Pred Class {c}")
        axes[1, c-1].imshow(target_seg[0, slice_idx].cpu() == c, cmap='Greens')
        axes[1, c-1].set_title(f"GT Class {c}")
        for ax in [axes[0, c-1], axes[1, c-1]]:
            ax.axis('off')
    plt.suptitle("Center Slice Prediction vs Ground Truth")
    plt.tight_layout()
    plt.show()
