"""YogaNet: temporal multi-task network over a short window of pose features.

Input  : (B, T, 39) per-frame features (normalised keypoints + joint angles)
Heads  : 1. pose class (4 yoga poses + 'other')            softmax
         2. correct vs incorrect (for the target pose)        sigmoid
         3. error types (multi-label, conditioned on pose)    sigmoid
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ERRORS, NUM_ERR, NUM_POSES, POSE_ERRORS, POSES
from .geometry import NUM_FEATURES


def applicability_matrix() -> torch.Tensor:
    """(NUM_POSES, NUM_ERR) 1 where an error type is meaningful for the pose."""
    A = np.zeros((NUM_POSES, NUM_ERR), np.float32)
    for pi, pose in enumerate(POSES):
        for code in POSE_ERRORS.get(pose, []):
            A[pi, ERRORS.index(code)] = 1
    return torch.from_numpy(A)


class YogaNet(nn.Module):
    def __init__(self, in_dim: int = NUM_FEATURES, hid: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hid=hid, dropout=dropout)
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.GELU())
        self.conv1 = nn.Conv1d(hid, hid, 3, padding=1)
        self.conv2 = nn.Conv1d(hid, hid, 3, padding=1)
        self.norm = nn.LayerNorm(hid)
        self.gru = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
        self.trunk = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Dropout(dropout))
        self.pose_head = nn.Linear(hid, NUM_POSES)
        self.correct_head = nn.Linear(hid, 1)
        self.err_head = nn.Sequential(nn.Linear(hid + NUM_POSES, hid), nn.GELU(),
                                      nn.Linear(hid, NUM_ERR))
        self.register_buffer("applic", applicability_matrix())

    def forward(self, x: torch.Tensor):
        h = self.inp(x)                                   # (B,T,H)
        c = h.transpose(1, 2)
        c = F.gelu(self.conv1(c))
        c = F.gelu(self.conv2(c))
        h = self.norm(h + c.transpose(1, 2))              # residual temporal conv
        g, _ = self.gru(h)                                # (B,T,H)
        pooled = torch.cat([g.mean(1), g.max(1).values], dim=-1)
        z = self.trunk(pooled)
        pose_logits = self.pose_head(z)
        correct_logit = self.correct_head(z).squeeze(-1)
        err_logits = self.err_head(torch.cat([z, F.softmax(pose_logits, -1)], dim=-1))
        return pose_logits, correct_logit, err_logits


def multitask_loss(out, pose, errors, is_target, applic, w=(1.0, 0.5, 1.0)):
    pose_logits, correct_logit, err_logits = out
    l_pose = F.cross_entropy(pose_logits, pose, label_smoothing=0.05)
    n = is_target.sum().clamp(min=1.0)
    correct = (errors.sum(-1) == 0).float()
    l_corr = (F.binary_cross_entropy_with_logits(correct_logit, correct, reduction="none")
              * is_target).sum() / n
    mask = applic[pose]                                    # (B,E) true pose applicability
    l_err = (F.binary_cross_entropy_with_logits(err_logits, errors, reduction="none")
             * mask).sum() / mask.sum().clamp(min=1.0)
    return w[0] * l_pose + w[1] * l_corr + w[2] * l_err, (l_pose.item(), l_corr.item(), l_err.item())


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
