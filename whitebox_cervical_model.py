# -*- coding: utf-8 -*-
"""
whitebox_cervical_model.py (Version X)
Full-image white-box: explicit pathological concepts → energy E → closed-form attention A=softmax(-E/τ).
Attention is used only for interpretable feature sampling: weighted readout from the last backbone feature map.
The final classifier head is a linear layer (no concept head, no residual head).

Usage (self-check):
  python whitebox_cervical_model.py
"""

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# Small helpers
# ==============================

def _force_odd(k: int) -> int:
    """Force kernel size to be odd (e.g., 10 -> 11)."""
    return k if (k % 2 == 1) else (k + 1)

def _repeat_depthwise_kernel(k: torch.Tensor, C: int) -> torch.Tensor:
    # k: [1,1,kh,kw] -> [C,1,kh,kw] depthwise per-channel identical kernels
    return k.repeat(C, 1, 1, 1)

def _to_gray_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] if normalized that way; map to [0,1]
    if x.shape[1] == 3:
        r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
        y = 0.299*r + 0.587*g + 0.114*b
    else:
        y = x
    y = 0.5*(y + 1.0)
    return y.clamp(0.0, 1.0)

def _otsu(gray: torch.Tensor, bins: int = 32) -> torch.Tensor:
    B, _, H, W = gray.shape
    edges = torch.linspace(0, 1, bins+1, device=gray.device)
    xs = torch.linspace(0, 1, bins,    device=gray.device)
    thrs = []
    for b in range(B):
        v = gray[b].reshape(-1)
        h = torch.histc(v, bins=bins, min=0.0, max=1.0)
        p = h / (H*W + 1e-6)
        omega = torch.cumsum(p, dim=0)
        mu    = torch.cumsum(p * xs, dim=0)
        mu_t  = mu[-1]
        sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1-omega) + 1e-8)
        k = torch.argmax(sigma_b2)
        thrs.append(edges[k])
    return torch.stack(thrs).view(B,1,1,1)

def _morph_open_close(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    k = _force_odd(int(k))
    pad = k//2
    # opening
    erode  = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    dilate =  F.max_pool2d(erode, kernel_size=k, stride=1, padding=pad)
    # closing
    dilate =  F.max_pool2d(dilate, kernel_size=k, stride=1, padding=pad)
    erode2 = -F.max_pool2d(-dilate, kernel_size=k, stride=1, padding=pad)
    return (erode2>0.5).float()

def _dilate(mask: torch.Tensor, r: int = 8) -> torch.Tensor:
    """
    Morphological dilation with odd kernel size.
    r: dilation radius (interpreted as kernel size; will be forced to odd).
    """
    k = _force_odd(int(r))
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)

def _local_sum(x: torch.Tensor, k: int = 9) -> torch.Tensor:
    """
    Local sum pooling with odd kernel size (same spatial size).
    """
    k = _force_odd(int(k))
    pad = k // 2
    ones = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype)
    return F.conv2d(x, ones, stride=1, padding=pad)

def _sobel(gray: torch.Tensor):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    ang = torch.atan2(gy, gx)
    return mag, ang

# ==============================
# Analytic backbone (white-box-ish)
# ==============================

def _softplus_inv(y: float) -> float:
    return float(math.log(math.exp(y) - 1.0))

class NearDiagNonNegMix(nn.Module):
    """
    Non-negative near-diagonal 1x1 mixing (square):
      W_eff = softplus(P) ⊙ Mask + eps * I
    """
    def __init__(self, dim: int, bandwidth: int = 1, eps: float = 1e-3,
                 offdiag_init: float = 0.018, diag_init: float = 1.0):
        super().__init__()
        self.in_dim = dim
        self.out_dim = dim
        self.bandwidth = int(max(0, bandwidth))
        self.eps = float(eps)

        self.P = nn.Parameter(torch.zeros(dim, dim))

        idx = torch.arange(dim)
        i = idx.view(-1, 1)
        j = idx.view( 1,-1)
        band = (torch.abs(i - j) <= self.bandwidth).float()
        self.register_buffer('mask', band)
        eye = torch.eye(dim)
        self.register_buffer('eye', eye)

        with torch.no_grad():
            self.P.fill_(_softplus_inv(1e-8))
            self.P.diagonal().copy_(torch.full((dim,), _softplus_inv(diag_init)))
            if self.bandwidth > 0 and offdiag_init > 0:
                offval = _softplus_inv(offdiag_init)
                offmask = (self.mask - self.eye).bool()
                self.P[offmask] = offval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = F.softplus(self.P) * self.mask + self.eps * self.eye
        W = W.view(self.out_dim, self.in_dim, 1, 1)
        return F.conv2d(x, W, bias=None, stride=1, padding=0)

class NearDiagNonNegMixIO(nn.Module):
    """
    Non-negative near-diagonal 1x1 mixing (rectangular in/out).
    """
    def __init__(self, in_dim: int, out_dim: int, bandwidth: int = 1, eps: float = 1e-3,
                 offdiag_init: float = 0.018, diag_init: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bandwidth = int(max(0, bandwidth))
        self.eps = float(eps)

        self.P = nn.Parameter(torch.zeros(out_dim, in_dim))

        m = torch.zeros(out_dim, in_dim)
        L = min(in_dim, out_dim)
        out_pos = torch.linspace(0, out_dim - 1, steps=L)
        in_pos  = torch.linspace(0, in_dim  - 1, steps=L)
        for k in range(L):
            o = int(round(float(out_pos[k])))
            i = int(round(float(in_pos[k])))
            o0, o1 = max(0, o - self.bandwidth), min(out_dim - 1, o + self.bandwidth)
            i0, i1 = max(0, i - self.bandwidth), min(in_dim  - 1, i + self.bandwidth)
            m[o0:o1+1, i0:i1+1] = 1.0
        self.register_buffer('mask', m)

        eye = torch.zeros(out_dim, in_dim)
        for k in range(L):
            o = int(round(float(out_pos[k]))); i = int(round(float(in_pos[k])))
            eye[o, i] = 1.0
        self.register_buffer('eye', eye)

        with torch.no_grad():
            self.P.fill_(_softplus_inv(1e-8))
            diag_vals = torch.full((L,), _softplus_inv(diag_init))
            for k in range(L):
                o = int(round(float(out_pos[k])))
                i = int(round(float(in_pos[k])))
                self.P[o, i] = diag_vals[k]
            if self.bandwidth > 0 and offdiag_init > 0:
                offval = _softplus_inv(offdiag_init)
                band_only = (self.mask - self.eye).bool()
                self.P[band_only] = offval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = F.softplus(self.P) * self.mask + self.eps * self.eye
        W = W.view(self.out_dim, self.in_dim, 1, 1)
        return F.conv2d(x, W, bias=None, stride=1, padding=0)

class HybridReparamConvBN(nn.Module):
    """
    Hybrid reparameterized downsampling:
      y = BN( (1 - σ(g)) * Conv3x3(x) + σ(g) * Mix2( DW_k( Mix1(x) ) ) )
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, with_bn=True, bandwidth: int = 1, eps: float = 1e-3):
        super().__init__()
        self.conv_orig = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                                   dilation=dilation, bias=False)
        self.mix1 = NearDiagNonNegMix(in_planes, bandwidth=bandwidth, eps=eps)
        self.dw   = nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding,
                              dilation=dilation, groups=in_planes, bias=False)
        self.mix2 = NearDiagNonNegMixIO(in_planes, out_planes, bandwidth=bandwidth, eps=eps)

        self.bn = nn.BatchNorm2d(out_planes) if with_bn else nn.Identity()
        self.gate_p = nn.Parameter(torch.tensor(-6.0))  # σ≈0 at init

        nn.init.kaiming_normal_(self.conv_orig.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.dw.weight,        mode="fan_out", nonlinearity="relu")
        if isinstance(self.bn, nn.BatchNorm2d):
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_orig = self.conv_orig(x)
        y_rep  = self.mix2(self.dw(self.mix1(x)))
        gate = torch.sigmoid(self.gate_p)
        y = (1.0 - gate) * y_orig + gate * y_rep
        return self.bn(y)

class AnalyticBlock(nn.Module):
    """
    Shape-preserving analytic block:
      - fixed Sobel/Laplacian per-channel
      - non-negative learned scalars to combine them
      - near-diagonal non-negative 1x1 mix + BN
      - convex residual
    """
    def __init__(self, dim: int):
        super().__init__()
        self.C = dim
        sx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[ 1, 2, 1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)
        self.register_buffer('laplace', lap)

        self.alpha_gm  = nn.Parameter(torch.tensor(1.0))
        self.alpha_lap = nn.Parameter(torch.tensor(0.5))

        self.proj = NearDiagNonNegMix(dim, bandwidth=1, eps=1e-3, offdiag_init=0.018, diag_init=1.0)
        self.bn   = nn.BatchNorm2d(dim)
        self.gamma_p = nn.Parameter(torch.tensor(0.5))
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias,   0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        assert C == self.C
        w_sx  = _repeat_depthwise_kernel(self.sobel_x,  C)
        w_sy  = _repeat_depthwise_kernel(self.sobel_y,  C)
        w_lap = _repeat_depthwise_kernel(self.laplace,  C)

        gx = F.conv2d(x, w_sx, padding=1, groups=C)
        gy = F.conv2d(x, w_sy, padding=1, groups=C)
        gm = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = torch.abs(F.conv2d(x, w_lap, padding=1, groups=C))

        a_gm  = F.softplus(self.alpha_gm)
        a_lap = F.softplus(self.alpha_lap)
        feat = F.relu(a_gm * gm + a_lap * lap, inplace=True)

        x_hat = self.bn(self.proj(feat))
        gamma = torch.sigmoid(self.gamma_p)
        return x + gamma * (x_hat - x)

class AnalyticBackbone(nn.Module):
    """
    Returns two things:
      - fmap: last feature map [B, D, H', W']
      - pooled: global average pooled vector [B, D] (reserved)
    """
    def __init__(self, base_dim=64, depths=(1, 1, 1, 1)):
        super().__init__()
        self.in_ch = 32
        self.stem = nn.Sequential(
            HybridReparamConvBN(3, self.in_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(inplace=True)
        )
        self.stages = nn.ModuleList()
        for i, d in enumerate(depths):
            embed_dim = base_dim * (2 ** i)
            down = HybridReparamConvBN(self.in_ch, embed_dim, kernel_size=3, stride=2, padding=1)
            self.in_ch = embed_dim
            blocks = [AnalyticBlock(self.in_ch) for _ in range(d)]
            self.stages.append(nn.Sequential(down, *blocks))
        self.norm = nn.BatchNorm2d(self.in_ch)
        self.avg  = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.norm(x)
        return x  # fmap [B, D, H', W']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.forward_features(x)            # [B, D, H', W']
        pooled = self.avg(fmap).flatten(1)         # [B, D]
        return pooled

def build_default_backbone() -> AnalyticBackbone:
    return AnalyticBackbone(base_dim=64, depths=(1,1,1,1))

# ==============================
# Full-image explicit concepts & energy attention
# ==============================

def compute_concept_maps(image_rgb: torch.Tensor, cell_dilate: int = 10, k_local: int = 9):
    """
    image_rgb: [B,3,H,W] in range [-1,1] (after Normalize mean=0.5,std=0.5)
    Return z_maps: [B,4,H,W] = [roughness, curvature_var, N/C, local_entropy_like]
    """
    B, C, H, W = image_rgb.shape
    device = image_rgb.device
    gray = _to_gray_01(image_rgb)  # [B,1,H,W] in [0,1]
    thr  = _otsu(gray)

    # nucleus mask + cleaning
    nuc = (gray <= thr).float()
    nuc = _morph_open_close(nuc, k=3)

    # boundary & roughness map
    er = -F.max_pool2d(-nuc, kernel_size=3, stride=1, padding=1)
    boundary = (nuc - er).clamp_min(0.0)

    lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=gray.dtype, device=device).view(1,1,3,3)
    lap = F.conv2d(nuc, lap_k, padding=1).abs()
    # normalize by local boundary support to avoid division by tiny counts
    k_norm = _force_odd(5)
    den = _local_sum(boundary, k=k_norm) + 1e-6
    roughness = (lap * boundary) / den  # [B,1,H,W]

    # curvature variance (circular variance on local gradients)
    mag, ang = _sobel(gray)           # [B,1,H,W]
    sb, cb = torch.sin(ang), torch.cos(ang)
    ms = _local_sum(sb, k=k_local)    # local sums
    mc = _local_sum(cb, k=k_local)
    den_orient = _local_sum(torch.ones_like(gray), k=k_local) + 1e-6
    ms, mc = ms/den_orient, mc/den_orient
    circ_var = 1.0 - torch.sqrt(ms**2 + mc**2 + 1e-6)
    curvature = circ_var  # [B,1,H,W]

    # N/C ratio map (local)
    cell = _dilate(nuc, r=cell_dilate).clamp(max=1.0)
    A_n = _local_sum(nuc,  k=k_local)
    A_c = (_local_sum(cell, k=k_local) - A_n).clamp_min(1.0)
    nc  = A_n / A_c  # [B,1,H,W]

    # Local "entropy-like" via local histogram surrogate
    g2 = _local_sum(gray*gray, k=k_local) / den_orient
    g1 = _local_sum(gray,      k=k_local) / den_orient
    var_local = (g2 - g1*g1).clamp_min(0.0)
    ent_like = torch.log1p(var_local)

    z_maps = torch.cat([roughness, curvature, nc, ent_like], dim=1)  # [B,4,H,W]
    return z_maps, nuc

class MonotoneCalibrator(nn.Module):
    """Monotone increasing linear calibrator: f(z) = softplus(a) * z + b"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a) * z + self.b

class EnergyAttentionImage(nn.Module):
    """
    z_maps: [B,4,H,W] → per-channel monotone calibration → simplex α weighting → E.
    A = softmax(-E/τ) (closed-form from negative free energy).
    Returns A:[B,1,Ht,Wt], z_maps, nucleus mask, and resized energy.
    """
    def __init__(self, num_terms: int = 4, cell_dilate: int = 10, k_local: int = 9):
        super().__init__()
        self.num_terms = num_terms
        self.cell_dilate = cell_dilate
        self.k_local = k_local
        self.cals = nn.ModuleList([MonotoneCalibrator() for _ in range(num_terms)])
        self.alpha_logits = nn.Parameter(torch.zeros(num_terms))  # simplex over concepts
        self.tau_p = nn.Parameter(torch.tensor(0.5))              # tau = 0.2 + softplus(tau_p)

    def forward(self, image_rgb: torch.Tensor, target_hw: Tuple[int,int]):
        Ht, Wt = target_hw
        # 1) per-pixel concept maps on full image
        z_maps, nuc = compute_concept_maps(image_rgb, cell_dilate=self.cell_dilate, k_local=self.k_local)  # [B,4,H,W]

        # 2) monotone calibration + numerically stable clipping
        z0 = torch.log1p(z_maps[:,0:1].clamp_min(0))            # roughness
        z1 = z_maps[:,1:2].clamp(0.0, 1.0)                      # curvature variance in [0,1]
        z2 = torch.log1p(z_maps[:,2:3].clamp_min(0))            # nc ratio
        z3 = z_maps[:,3:4].clamp(0.0, math.log(32.0))           # entropy-like bounded
        z_stable = torch.cat([z0, z1, z2, z3], dim=1)           # [B,4,H,W]

        z_cal = torch.cat([ self.cals[i](z_stable[:, i:i+1]) for i in range(self.num_terms) ], dim=1)  # [B,4,H,W]

        # 3) energy E and attention A on target grid
        alpha = torch.softmax(self.alpha_logits, dim=0).view(1, self.num_terms, 1, 1)  # [1,4,1,1]
        E = (alpha * z_cal).sum(dim=1, keepdim=True)  # [B,1,H,W]

        if (E.shape[2], E.shape[3]) != (Ht, Wt):
            E_t = F.interpolate(E, size=(Ht, Wt), mode='bilinear', align_corners=False)
        else:
            E_t = E

        tau = 0.2 + F.softplus(self.tau_p)
        B, _, Hh, Wh = E_t.shape
        A = torch.softmax((-E_t / tau).view(B, 1, Hh*Wh), dim=-1).view(B, 1, Hh, Wh)  # [B,1,Ht,Wt]

        return A, z_maps, nuc, E_t

# ==============================
# Top-level network (Version X)
# ==============================

class WhiteBoxEnergyNet(nn.Module):
    """
    Decision path (single path):
      x → backbone last feature map fmap[B,D,H',W'] →
      energy-derived attention A[1,H',W'] → weighted readout m[B,D] → Linear[B,num_class]
    """
    def __init__(self, num_class: int,
                 cell_dilate: int = 10,
                 k_local: int = 9):
        super().__init__()
        self.num_class = num_class

        # backbone -> feature map
        self.backbone = build_default_backbone()
        self.feat_dim = self.backbone.in_ch

        # white-box energy attention (full image) used only for interpretable sampling
        self.energy_att = EnergyAttentionImage(num_terms=4, cell_dilate=cell_dilate, k_local=k_local)

        # linear classification head (only decision head)
        self.classifier = nn.Linear(self.feat_dim, num_class)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)

    @torch.no_grad()
    def _qc_stats(self, z_maps: torch.Tensor, nuc: torch.Tensor, A: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simple QC stats: global means and attention-weighted means (examples).
        z_maps: [B,4,H,W], nuc: [B,1,H,W], A: [B,1,Ht,Wt] (aligned to fmap).
        Note: assumes A is aligned with fmap (we use fmap's H',W' in forward).
        """
        B, _, H, W = z_maps.shape
        # downsample z_maps to A resolution, then A-weighted mean
        _, _, Ht, Wt = A.shape
        z_t = F.interpolate(z_maps, size=(Ht, Wt), mode='bilinear', align_corners=False)
        A1 = A / (A.sum(dim=(2,3), keepdim=True) + 1e-6)
        amean = (z_t * A1).sum(dim=(2,3))  # [B,4]
        gmean = z_maps.flatten(2).mean(dim=2)  # [B,4]
        out = {
            "mean_att_roughness": amean[:,0],
            "mean_att_curvvar":   amean[:,1],
            "mean_att_nc":        amean[:,2],
            "mean_att_entropy":   amean[:,3],
            "mean_global_roughness": gmean[:,0],
            "mean_global_curvvar":   gmean[:,1],
            "mean_global_nc":        gmean[:,2],
            "mean_global_entropy":   gmean[:,3],
        }
        return out

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # 1) backbone features
        fmap = self.backbone.forward_features(x)              # [B, D, H', W']
        B, D, Ht, Wt = fmap.shape

        # 2) energy-derived attention A aligned to (H',W')
        A, z_maps, nuc, E_t = self.energy_att(x, target_hw=(Ht, Wt))  # A:[B,1,H',W']

        # 3) attention-weighted readout (A is a probability distribution, sums to 1)
        A1 = A  # already softmax probabilities
        m = (fmap * A1).flatten(2).sum(dim=2)                 # [B, D]

        # 4) linear classification
        logits = self.classifier(m)                           # [B, num_class]

        if not return_aux:
            return logits
        else:
            aux = {
                "A": A,                 # [B,1,H',W']
                "z_maps": z_maps,       # [B,4,H,W]
                "nuc": nuc,             # [B,1,H,W]
                "energy": E_t,          # [B,1,H',W']
                "qc_stats": self._qc_stats(z_maps, nuc, A)
            }
            return logits, aux

# ==============================
# Sanity test
# ==============================

if __name__ == "__main__":
    torch.manual_seed(0)
    m = WhiteBoxEnergyNet(num_class=5, cell_dilate=10, k_local=10)  # even k_local is OK; forced to odd internally
    x = torch.randn(2, 3, 224, 224)  # mimic normalized [-1,1] images after (0.5,0.5,0.5) mean/std
    y = m(x, return_aux=True)
    if isinstance(y, tuple):
        logits, aux = y
        print("logits:", logits.shape)
        print("A:", aux["A"].shape, "z_maps:", aux["z_maps"].shape, "energy:", aux["energy"].shape)
        keys = list(aux["qc_stats"].keys())
        print("QC keys sample:", keys[:4])
    else:
        print("logits:", y.shape)
