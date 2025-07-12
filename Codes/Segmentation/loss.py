import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, TverskyLoss
from monai.utils import LossReduction
import numpy as np


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
        loss_weights=(0.4, 0.4, 0.2),  # (dice_ce, tversky, surface)
        max_weight_factor=2.5,
        surface_loss_max_dist=10.0,
        boundary_kernel_size=3,
        subsample_boundaries=None  # Set to int > 0 to subsample boundary points
    ):
        """
        Volume-aware multi-component loss function for 3D medical image segmentation.
        
        Args:
            include_background: Whether to include background class in loss calculation
            softmax: Apply softmax to prediction before loss calculation
            tversky_alpha: Alpha parameter for Tversky loss (FP weight)
            tversky_beta: Beta parameter for Tversky loss (FN weight)
            class_weights: Static weights per class [background, class1, class2, ...]
            baseline_volumes: Expected volumes per class in voxels [background, class1, ...]
            epsilon: Small value to avoid division by zero
            reduction: How to reduce loss over batch ("mean" or "none")
            loss_weights: Weights for (dice_ce, tversky, surface) components
            max_weight_factor: Maximum multiplier for dynamic weights
            surface_loss_max_dist: Maximum distance penalty for surface loss
            boundary_kernel_size: Size of erosion kernel for boundary detection
            subsample_boundaries: If set, subsample boundary points to this number for efficiency
        """
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.epsilon = epsilon
        self.reduction = reduction
        self.loss_weights = loss_weights
        self.max_weight_factor = max_weight_factor
        self.surface_loss_max_dist = surface_loss_max_dist
        self.boundary_kernel_size = boundary_kernel_size
        self.subsample_boundaries = subsample_boundaries

        # Default weights and baseline volumes if not provided
        cw = class_weights or [1.0, 2.0, 1.5, 2.5, 1.5]
        bv = baseline_volumes or [0.0, 2000.0, 8000.0, 1000.0, 3000.0]
        self.n_classes = len(cw)

        # Component losses
        self.dice_ce = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            lambda_dice=0.5,
            lambda_ce=0.5,
            reduction=LossReduction.MEAN,  # Changed to get per-batch losses
        )
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            alpha=tversky_alpha,
            beta=tversky_beta,
            reduction=LossReduction.MEAN,  # Changed to get per-batch losses
        )

        # Register buffers for class weights and baseline volumes
        self.register_buffer("static_w", torch.tensor(cw, dtype=torch.float32))
        self.register_buffer("baseline_w", torch.tensor(bv, dtype=torch.float32))
        
        # Create boundary detection kernel once
        kernel_size = self.boundary_kernel_size
        self.register_buffer(
            "erosion_kernel", 
            torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32)
        )

    def _compute_dynamic_weights(self, onehot):
        """Compute dynamic weights based on class volumes"""
        # Calculate volumes per class [B, C]
        vols = onehot.sum(dim=(2, 3, 4))
        
        # Get baseline volumes and static weights on correct device
        baseline = self.baseline_w.to(onehot.device)
        static = self.static_w.to(onehot.device)

        # Calculate volume-based weight multipliers
        # Higher weight for smaller-than-expected classes
        with torch.no_grad():
            mult = torch.sqrt(baseline / torch.clamp(vols, min=self.epsilon))
            mult = torch.clamp(mult, max=self.max_weight_factor)
            
            # Apply static weights
            eff_weights = static * mult
            
            # Normalize by maximum weight to keep scale reasonable
            normalized = eff_weights / torch.clamp(
                eff_weights.max(dim=1, keepdim=True)[0], 
                min=self.epsilon
            )
            
        return normalized

    def _extract_boundaries(self, binary_mask):
        """
        Extract boundary points from a binary mask using erosion
        
        Args:
            binary_mask: (B, 1, D, H, W) tensor of binary mask
            
        Returns:
            Tuple of (boundary_mask, boundary_points)
            boundary_mask: (B, D, H, W) boolean tensor
            boundary_points: list of (N, 3) coordinate tensors or None if empty
        """
        # Erode with conv3d: we check where the sum under the kernel == kernel size
        kernel = self.erosion_kernel.to(binary_mask.device)
        kernel_size = kernel.numel()
        
        eroded = (F.conv3d(binary_mask, kernel, padding=self.boundary_kernel_size//2) == kernel_size).float()
        
        # Boundary = mask XOR eroded(mask)
        boundary = (binary_mask - eroded).abs().bool().squeeze(1)  # (B, D, H, W)
        
        # Extract coordinate points for each batch
        batch_size = binary_mask.shape[0]
        boundary_points = []
        
        for b in range(batch_size):
            if not boundary[b].any():
                boundary_points.append(None)
                continue
                
            # Get coordinates as (N, 3) tensor
            coords = boundary[b].nonzero(as_tuple=False).float()  # (N, 3)
            
            # Subsample points if requested and if we have enough points
            if self.subsample_boundaries and coords.shape[0] > self.subsample_boundaries:
                idx = torch.randperm(coords.shape[0], device=coords.device)[:self.subsample_boundaries]
                coords = coords[idx]
                
            boundary_points.append(coords)
            
        return boundary, boundary_points

    def compute_surface_loss(self, pred_probs, onehot_target, class_weights):
        """
        Compute surface distance loss between prediction and target boundaries
        
        Args:
            pred_probs: (B, C, D, H, W) float probabilities
            onehot_target: (B, C, D, H, W) one-hot encoded target
            class_weights: (B, C) class weights
            
        Returns:
            (B,) tensor of surface losses per batch
        """
        B, C, D, H, W = pred_probs.shape
        device = pred_probs.device
        
        # Threshold probabilities to get binary predictions
        pred_bin = (pred_probs > 0.5).float()
        gt_bin = onehot_target.float()
        
        # Initialize batch losses
        batch_losses = torch.zeros(B, device=device)
        
        for b in range(B):
            sample_loss = 0.0
            valid_classes = 0
            
            for c in range(C):
                # Skip background if not included
                if not self.include_background and c == 0:
                    continue
                
                # Get binary masks for this class
                pred_mask = pred_bin[b:b+1, c:c+1]  # (1, 1, D, H, W)
                target_mask = gt_bin[b:b+1, c:c+1]
                
                # Skip if both prediction and target are empty
                if pred_mask.sum() == 0 and target_mask.sum() == 0:
                    continue
                
                # Extract boundaries
                _, pred_points = self._extract_boundaries(pred_mask)
                _, target_points = self._extract_boundaries(target_mask)
                
                # Handle edge cases
                if pred_points[0] is None and target_points[0] is None:
                    continue  # Both have no boundary
                    
                if pred_points[0] is None or target_points[0] is None:
                    # One has boundary, other doesn't - maximum penalty
                    sample_loss += class_weights[b, c] * self.surface_loss_max_dist
                    valid_classes += 1
                    continue
                
                # Compute distances between boundary points
                # From prediction to target
                p2t_dists = torch.cdist(pred_points[0], target_points[0])  # (P, T)
                t2p_dists = p2t_dists.t()  # (T, P)
                
                # Average shortest distances in both directions
                p2t_min = p2t_dists.min(dim=1)[0].mean()  # Mean over prediction points
                t2p_min = t2p_dists.min(dim=1)[0].mean()  # Mean over target points
                
                # Symmetric distance
                avg_dist = (p2t_min + t2p_min) / 2
                
                # Clamp any nan/inf values
                if not torch.isfinite(avg_dist):
                    avg_dist = torch.tensor(self.surface_loss_max_dist, device=device)
                
                # Add weighted distance for this class
                sample_loss += class_weights[b, c] * avg_dist
                valid_classes += 1
            
            # Normalize by number of valid classes to get average surface distance
            if valid_classes > 0:
                batch_losses[b] = sample_loss / valid_classes
            else:
                batch_losses[b] = torch.tensor(0.0, device=device)
        
        return batch_losses

    def forward(self, pred, target):
        """
        Compute the combined loss
        
        Args:
            pred: (B, C, D, H, W) model prediction logits
            target: (B, C, D, H, W) one-hot encoded target
            
        Returns:
            Dict containing loss components and final loss
        """
        assert pred.dim() == 5 and target.dim() == 5, f"Both pred and target must be 5-D, got {pred.shape} and {target.shape}"
        device = pred.device
        
        # Compute dynamic weights based on volumes
        dynamic_weights = self._compute_dynamic_weights(target)  # [B, C]
        
        # Calculate component losses
        # Set dynamic weights for dice_ce loss
        # Note: we need to clone weights to avoid modifying the original
        self.dice_ce.class_weight = dynamic_weights.clone()
        
        # Dice+CE loss (returns per-batch loss)
        dice_ce_loss = self.dice_ce(pred, target)  # [B]
        
        # Tversky loss with dynamic weights
        tversky_loss = self.tversky(pred, target)  # [B]
        
        # For surface loss, we need class probabilities
        probs = F.softmax(pred, dim=1) if self.softmax else pred
        
        # Surface distance loss
        surface_loss = self.compute_surface_loss(probs, target, dynamic_weights)
        
        # Combine losses with component weights
        w1, w2, w3 = self.loss_weights
        total_loss = w1 * dice_ce_loss + w2 * tversky_loss + w3 * surface_loss
        
        # Return based on reduction mode
        if self.reduction == "mean":
            return {
                "loss": total_loss.mean(),
                "dice_ce_loss": dice_ce_loss.mean(),
                "tversky_loss": tversky_loss.mean(),
                "surface_loss": surface_loss.mean(),
                "dynamic_weights": dynamic_weights,
                "per_sample_loss": total_loss,
            }
        else:
            return {
                "loss": total_loss,
                "dice_ce_loss": dice_ce_loss,
                "tversky_loss": tversky_loss,
                "surface_loss": surface_loss,
                "dynamic_weights": dynamic_weights,
                "per_sample_loss": total_loss,
            }


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced segmentation
    """
    def __init__(
        self, 
        include_background=False,
        to_onehot_y=False,
        softmax=True,
        alpha=0.3,
        beta=0.7,
        gamma=2.0,
        reduction="mean"
    ):
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        if self.softmax:
            pred = F.softmax(pred, dim=1)
            
        if self.to_onehot_y:
            target = F.one_hot(
                target.squeeze(1).long(), 
                num_classes=pred.shape[1]
            ).permute(0, 4, 1, 2, 3).float()
        
        # Flatten for easier calculation
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Calculate components for Tversky
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Tversky per class
        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        
        # Exclude background if needed
        if not self.include_background:
            tversky = tversky[:, 1:]
            
        # Apply focal power
        focal_tversky = (1 - tversky) ** self.gamma
        
        # Average over classes
        loss = focal_tversky.mean(dim=1)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        return loss


def get_brats_loss(loss_type="volume_aware", **kwargs):
    """
    Factory function to get loss function for BraTS 2025 segmentation
    
    Args:
        loss_type: Type of loss function to use
            - "volume_aware": VolumeAwareLoss with default settings
            - "focal_tversky": FocalTverskyLoss
            - "combined": VolumeAwareLoss with custom settings
        **kwargs: Additional parameters to pass to the loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == "volume_aware":
        return VolumeAwareLoss(**kwargs)
    elif loss_type == "focal_tversky":
        return FocalTverskyLoss(**kwargs)
    elif loss_type == "combined":
        # Example of custom settings
        default_params = {
            "include_background": False,
            "softmax": True,
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "class_weights": [1.0, 2.0, 1.5, 2.5, 1.5],  # Background, NETC, SNFH, ET, RC
            "baseline_volumes": [0.0, 2000.0, 8000.0, 1000.0, 3000.0],
            "loss_weights": (0.35, 0.35, 0.3),  # Increased surface loss weight
            "max_weight_factor": 3.0,  # Allow more aggressive weighting
            "subsample_boundaries": 2000,  # Subsample boundaries for efficiency
        }
        default_params.update(kwargs)
        return VolumeAwareLoss(**default_params)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


