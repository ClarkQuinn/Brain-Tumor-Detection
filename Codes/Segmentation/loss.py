import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, TverskyLoss
from monai.networks.utils import one_hot
from monai.utils import LossReduction
from monai.metrics.utils import get_surface_distance
from monai.metrics import compute_average_surface_distance
import numpy as np
import scipy.ndimage as nd




class VolumeAwareLoss(nn.Module):
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
   ):
       super().__init__()
       self.include_background = include_background
       self.softmax = softmax
       self.epsilon = epsilon
       self.reduction = reduction


       cw = class_weights or [1.0, 2.0, 1.5, 2.5, 1.5]
       bv = baseline_volumes or [0.0, 2000.0, 8000.0, 1000.0, 3000.0]
       self.n_classes = len(cw)


       # Always expect one-hot inputs => disable MONAI's internal one-hot
       self.dice_ce = DiceCELoss(
           include_background=include_background,
           to_onehot_y=False,
           softmax=softmax,
           lambda_dice=0.5,
           lambda_ce=0.5,
           reduction=LossReduction.MEAN,
       )
       self.tversky = TverskyLoss(
           include_background=include_background,
           to_onehot_y=False,
           softmax=softmax,
           alpha=tversky_alpha,
           beta=tversky_beta,
           reduction=LossReduction.MEAN,
       )


       self.register_buffer("static_w", torch.tensor(cw, dtype=torch.float32))
       self.register_buffer("baseline_w", torch.tensor(bv, dtype=torch.float32))


   def _compute_dynamic_weights(self, onehot):
       vols = onehot.sum(dim=(2,3,4))                               # [B, C]
       mult = torch.sqrt(self.baseline_w / torch.clamp(vols, min=self.epsilon))
       mult = torch.clamp(mult, max=2.5)
       eff = self.static_w * mult                                  # [C]
       norm = eff.sum(dim=0, keepdim=True) / self.n_classes        # [1, C]
       return eff / torch.clamp(eff.max(dim=0, keepdim=True)[0], min=self.epsilon)



   def compute_surface_loss(
       self,
       pred_probs,        # [B, C, D, H, W], softmaxed probabilities
       onehot_target,     # [B, C, D, H, W], your one-hot GT
       weights            # [C], normalized class weights
   ):
       """
       Returns a tensor of shape [B], the per‐sample weighted surface loss.
       """
       B, C, D, H, W = pred_probs.size()
       losses = torch.zeros(B, device=pred_probs.device)


       # binarize predictions at 0.5 threshold
       pred_bin = (pred_probs > 0.5).cpu().numpy()


       # move onehot_target to CPU numpy for boundary ops
       gt_bin = onehot_target.cpu().numpy().astype(bool)


       for b in range(B):
           sample_loss = 0.0
           for c in range(C):
               if not self.include_background and c == 0:
                   continue


               pm = pred_bin[b, c]
               tm = gt_bin[b, c]


               # Skip if both are all zeros (no region present)
               if not pm.any() and not tm.any():
                   continue


               # compute boundary voxels via erosion difference
               struct = nd.generate_binary_structure(3, 1)  # 3D, connectivity=1
               pm_eroded = nd.binary_erosion(pm, structure=struct)
               tm_eroded = nd.binary_erosion(tm, structure=struct)
               pm_bd = pm ^ pm_eroded
               tm_bd = tm ^ tm_eroded


               print(f"Sample {b}, Class {c}: pred boundary voxels = {pm_bd.sum()}, gt boundary voxels = {tm_bd.sum()}")


               # if neither has boundary, skip
               if not pm_bd.any() and not tm_bd.any():
                   continue


               # if one has no boundary at all, penalize heavily
               if not pm_bd.any() or not tm_bd.any():
                   sample_loss += weights[c] * 10.0
                   continue


               # Only compute surface distance if both masks have at least one voxel
               if not pm.any() or not tm.any():
                   continue  # skip surface distance computation to avoid warnings


               avg_d = compute_average_surface_distance(np.expand_dims(pm, 0), np.expand_dims(tm, 0))
               val = float(avg_d[0, 0].item() if hasattr(avg_d[0, 0], 'item') else avg_d[0, 0])
               if np.isnan(val) or np.isinf(val):
                   val = 10.0  # penalty for nan surface distance
               sample_loss += weights[c] * val


           # normalize by number of considered classes
           nc = C if self.include_background else (C - 1)
           losses[b] = sample_loss / nc


       return losses  # [B]


   def forward(self, pred, target):
       """
       pred   : [B, C, D, H, W] logits
       target : [B, C, D, H, W] one-hot
       """
       assert pred.dim()==5 and target.dim()==5, f"Both pred and target must be 5-D, got {pred.shape} and {target.shape}"


       # dynamic weights per class (same for all batch samples)
       W = self._compute_dynamic_weights(target)  # [C]


       # MONAI losses with reduction NONE
       dice_map = self.dice_ce(pred, target)    # [B, C]
       tversky_map = self.tversky(pred, target)    # [B, C]


       # Debug prints
       print("Dice map:", dice_map)
       print("Tversky map:", tversky_map)
       print("Dynamic weights:", W)


       # apply weights
       wd = (dice_map * W).sum(dim=1)  # [B]
       wt = (tversky_map * W).sum(dim=1)  # [B]     


       # compute surface loss
       probs = F.softmax(pred, dim=1) if self.softmax else pred
       ws = self.compute_surface_loss(
               probs,
               target,
               self.static_w
           )


       print("Surface loss:", ws)


       total = 0.4*wd + 0.4*wt + 0.2*ws


       if self.reduction=="mean":
           return {
               "loss":         total.mean(),
               "dice_ce_loss": wd.mean(),
               "tversky_loss": wt.mean(),
               "surface_loss": ws.mean(),
               "normalized_weights": W,
               "per_sample_loss": total,
           }
       else:
           return {
               "loss": total,
               "dice_ce_loss": wd,
               "tversky_loss": wt,
               "surface_loss": ws,
               "normalized_weights": W,
               "per_sample_loss": total,
           }




def get_brats_loss():
   return VolumeAwareLoss()


