"""
data/augmentation.py

Structure-aware augmentation model: encoder-decoder that blends a source
image with characteristics extracted from a reference image, while
preserving the underlying structure of the source.

Architecture:
  - Shared encoder (source branch + reference branch)
  - Structural feature extractor (edge/gradient aware)
  - Fusion module in latent space
  - Decoder (transposed convolutions + batch norm)
  - PSNR / SSIM quality gate before saving
"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

import config
from .preprocessing import quality_filter, scale_image


# ── sub-modules ────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    """Tiny helper: Conv2d -> BatchNorm2d -> ReLU."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _UpConvBnRelu(nn.Sequential):
    """Transposed conv for upsampling in the decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ImageEncoder(nn.Module):
    """
    Encodes a single image into a latent vector.

    Progression: 3 -> 32 -> 64 -> 128 channels, halving spatial dims each step.
    For 64x64 input the output feature map is 8x8x128 before flattening.
    """

    def __init__(self, in_channels=3, latent_dim=config.AUG_LATENT_DIM):
        super().__init__()
        self.block1 = _ConvBnRelu(in_channels, 32, stride=2)   # 64->32
        self.block2 = _ConvBnRelu(32, 64, stride=2)             # 32->16
        self.block3 = _ConvBnRelu(64, 128, stride=2)            # 16->8
        self.pool   = nn.AdaptiveAvgPool2d(1)                   # 8x8 -> 1x1
        self.fc     = nn.Linear(128, latent_dim)

        # keep the spatial feature map for the structural path
        self._spatial_feats = None

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        self._spatial_feats = x            # [B, 128, 8, 8] for structural use
        z = self.pool(x).flatten(1)        # [B, 128]
        z = self.fc(z)                     # [B, latent_dim]
        return z


class StructuralFeatureExtractor(nn.Module):
    """
    Lightweight Sobel-based structure extractor.  Runs on the raw source image
    to get edge/gradient information so the decoder can preserve object
    contours when blending reference style.

    We intentionally keep this gradient-based rather than learning it —
    structural geometry shouldn't be learned away by the network.
    """

    def __init__(self):
        super().__init__()
        # Fixed Sobel filters — not learned
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32)
        sobel_y = sobel_x.t()
        # apply same filter across all 3 colour channels
        kx = sobel_x.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        ky = sobel_y.view(1, 1, 3, 3).repeat(1, 3, 1, 1)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

        # small learned refinement on top
        self.refine = nn.Sequential(
            _ConvBnRelu(1, 16),
            _ConvBnRelu(16, 32),
        )

    def forward(self, x):
        # x: [B, 3, H, W], compute gradient magnitude per pixel
        gx = F.conv2d(x, self.kx, padding=1, groups=1)
        gy = F.conv2d(x, self.ky, padding=1, groups=1)
        # collapse channels → single magnitude map
        mag = (gx.pow(2) + gy.pow(2)).mean(dim=1, keepdim=True).sqrt()
        struct = self.refine(mag)          # [B, 32, H, W]
        return struct


class FusionModule(nn.Module):
    """
    Blends source latent vector z_src with reference latent vector z_ref.

    Simple but deliberate: element-wise addition after a small projection
    rather than concat+MLP — concat tends to make the decoder ignore the
    reference when data is scarce.
    """

    def __init__(self, latent_dim=config.AUG_LATENT_DIM):
        super().__init__()
        self.proj_src = nn.Linear(latent_dim, latent_dim)
        self.proj_ref = nn.Linear(latent_dim, latent_dim)
        self.gate     = nn.Sigmoid()

        # how much of the reference style to let through (learnable per dim)
        self.alpha    = nn.Parameter(torch.full((latent_dim,), 0.5))

    def forward(self, z_src, z_ref):
        ps = self.proj_src(z_src)
        pr = self.proj_ref(z_ref)
        blend_weight = self.gate(self.alpha)
        z_fused = blend_weight * pr + (1 - blend_weight) * ps
        return z_fused


class ImageDecoder(nn.Module):
    """
    Reconstructs an image from the fused latent vector.
    The structural feature map from the source encoder is concatenated at
    an intermediate layer to preserve object contours.
    """

    def __init__(self, latent_dim=config.AUG_LATENT_DIM, out_channels=3,
                 spatial_size=8):
        super().__init__()
        self.spatial_size = spatial_size

        self.fc = nn.Linear(latent_dim, 128 * spatial_size * spatial_size)
        self.bn = nn.BatchNorm1d(128 * spatial_size * spatial_size)

        # up1: 8->16, up2: 16->32, up3: 32->64
        self.up1 = _UpConvBnRelu(128 + 32, 64)    # +32 for structural feats (downsampled)
        self.up2 = _UpConvBnRelu(64, 32)
        self.up3 = _UpConvBnRelu(32, 16)

        self.out_conv = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z_fused, struct_feats):
        B = z_fused.size(0)
        x = F.relu(self.bn(self.fc(z_fused)))
        x = x.view(B, 128, self.spatial_size, self.spatial_size)

        # structural feats arrive at [B, 32, H, W]; pool to match 8x8
        s = F.adaptive_avg_pool2d(struct_feats, (self.spatial_size, self.spatial_size))
        x = torch.cat([x, s], dim=1)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out_conv(x)
        return x


# ── full augmentation model ────────────────────────────────────────────────

class StructureAwareAugmentor(nn.Module):
    """
    End-to-end encoder-decoder that generates a new image by combining:
      - object structure from the source image (preserved via Sobel feats)
      - visual style characteristics from the reference image

    Training uses a combination of reconstruction + perceptual-style losses
    so the decoder learns to produce plausible blends rather than just
    noisy averages.
    """

    def __init__(self, latent_dim=config.AUG_LATENT_DIM):
        super().__init__()
        self.encoder  = ImageEncoder(latent_dim=latent_dim)
        self.struct   = StructuralFeatureExtractor()
        self.fusion   = FusionModule(latent_dim=latent_dim)
        self.decoder  = ImageDecoder(latent_dim=latent_dim)

    def forward(self, src, ref):
        """
        src: [B, 3, H, W] — source (structure is kept from here)
        ref: [B, 3, H, W] — reference (style is borrowed from here)
        """
        z_src = self.encoder(src)
        z_ref = self.encoder(ref)

        struct_feats = self.struct(src)
        z_fused      = self.fusion(z_src, z_ref)
        generated    = self.decoder(z_fused, struct_feats)
        return generated

    def compute_loss(self, src, generated):
        """
        Pixel-level reconstruction loss + a simple gradient-magnitude
        consistency loss to encourage the model not to blur edges.
        """
        recon_loss = F.l1_loss(generated, src)

        # gradient consistency: generated edges should loosely follow source
        def gradient_map(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy

        dx_s, dy_s = gradient_map(src)
        dx_g, dy_g = gradient_map(generated)
        grad_loss = F.l1_loss(dx_g, dx_s) + F.l1_loss(dy_g, dy_s)

        return recon_loss + 0.1 * grad_loss


# ── dataset for augmentor training ────────────────────────────────────────

class _PairDataset(Dataset):
    """Randomly pairs source + reference from the same image folder."""

    def __init__(self, img_dir):
        self.paths = (sorted(Path(img_dir).glob("*.jpg")) +
                      sorted(Path(img_dir).glob("*.png")) +
                      sorted(Path(img_dir).glob("*.jpeg")))
        self.tf = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ref_idx = random.randint(0, len(self.paths) - 1)
        while ref_idx == idx:
            ref_idx = random.randint(0, len(self.paths) - 1)

        src = self.tf(Image.open(self.paths[idx]).convert("RGB"))
        ref = self.tf(Image.open(self.paths[ref_idx]).convert("RGB"))
        return src, ref, str(self.paths[idx])


# ── training helper ────────────────────────────────────────────────────────

def train_augmentor(img_dir, save_path=None, epochs=config.AUG_EPOCHS,
                    lr=config.AUG_LR, device=None):
    """Train the augmentor on images from img_dir and return the model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StructureAwareAugmentor().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds     = _PairDataset(img_dir)
    loader = DataLoader(ds, batch_size=config.AUG_BATCH_SIZE,
                        shuffle=True, num_workers=2)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for src, ref, _ in loader:
            src, ref = src.to(device), ref.to(device)
            opt.zero_grad()
            gen  = model(src, ref)
            loss = model.compute_loss(src, gen)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        if epoch % 10 == 0:
            print(f"[Augmentor] epoch {epoch}/{epochs}  loss={total/len(loader):.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Augmentor saved → {save_path}")

    return model


def generate_augmented_dataset(model, img_dir, out_dir,
                                device=None, filter_quality=True):
    """
    Run the trained augmentor over all images in img_dir.
    Generated images that don't clear PSNR/SSIM thresholds are skipped.
    Returns the number of accepted images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    inv_tf = transforms.Compose([
        transforms.Normalize(mean=[-1.0]*3, std=[2.0]*3),
        transforms.ToPILImage(),
    ])

    img_paths = (sorted(Path(img_dir).glob("*.jpg")) +
                 sorted(Path(img_dir).glob("*.png")) +
                 sorted(Path(img_dir).glob("*.jpeg")))

    model.eval()
    accepted = 0

    with torch.no_grad():
        for idx, path in enumerate(img_paths):
            ref_idx = random.randint(0, len(img_paths) - 1)
            while ref_idx == idx:
                ref_idx = random.randint(0, len(img_paths) - 1)

            src_pil = Image.open(path).convert("RGB")
            ref_pil = Image.open(img_paths[ref_idx]).convert("RGB")

            src_t = tf(src_pil).unsqueeze(0).to(device)
            ref_t = tf(ref_pil).unsqueeze(0).to(device)

            gen_t   = model(src_t, ref_t).squeeze(0).cpu()
            gen_pil = inv_tf(gen_t)

            if filter_quality:
                src_resized = src_pil.resize((config.IMG_SIZE, config.IMG_SIZE))
                if not quality_filter(src_resized, gen_pil):
                    continue

            out_path = os.path.join(out_dir, f"aug_{idx:04d}.png")
            gen_pil.save(out_path)
            accepted += 1

    print(f"Generated & accepted {accepted}/{len(img_paths)} images → {out_dir}")
    return accepted
