"""
tests/test_sanity.py

Sanity checks: verify all model components run without errors on dummy data.
Run with:  python -m pytest tests/  or  python tests/test_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

import config


# ── helpers ────────────────────────────────────────────────────────────────

def _make_batch(b=4, c=3, h=64, w=64):
    return torch.rand(b, c, h, w)


# ── augmentation model tests ────────────────────────────────────────────────

def test_image_encoder():
    from data.augmentation import ImageEncoder
    enc = ImageEncoder()
    src = _make_batch()
    z   = enc(src)
    assert z.shape == (4, config.AUG_LATENT_DIM), \
        f"Encoder output shape mismatch: {z.shape}"
    print("  [PASS] ImageEncoder")


def test_structural_extractor():
    from data.augmentation import StructuralFeatureExtractor
    ext  = StructuralFeatureExtractor()
    src  = _make_batch()
    out  = ext(src)
    assert out.shape[0] == 4, f"Batch dim wrong: {out.shape}"
    assert out.shape[1] == 32, f"Channel dim wrong: {out.shape}"
    print("  [PASS] StructuralFeatureExtractor")


def test_fusion_module():
    from data.augmentation import FusionModule
    fm   = FusionModule()
    z_s  = torch.rand(4, config.AUG_LATENT_DIM)
    z_r  = torch.rand(4, config.AUG_LATENT_DIM)
    z_f  = fm(z_s, z_r)
    assert z_f.shape == z_s.shape, f"Fusion output shape mismatch: {z_f.shape}"
    print("  [PASS] FusionModule")


def test_image_decoder():
    from data.augmentation import ImageDecoder, StructuralFeatureExtractor
    dec   = ImageDecoder()
    ext   = StructuralFeatureExtractor()
    src   = _make_batch()
    z_f   = torch.rand(4, config.AUG_LATENT_DIM)
    sf    = ext(src)
    out   = dec(z_f, sf)
    assert out.shape == (4, 3, 64, 64), f"Decoder output shape: {out.shape}"
    print("  [PASS] ImageDecoder")


def test_full_augmentor():
    from data.augmentation import StructureAwareAugmentor
    model = StructureAwareAugmentor()
    src   = _make_batch()
    ref   = _make_batch()
    gen   = model(src, ref)
    assert gen.shape == src.shape, f"Augmentor output shape: {gen.shape}"
    loss  = model.compute_loss(src, gen)
    assert loss.item() >= 0, "Loss should be non-negative"
    loss.backward()
    print("  [PASS] StructureAwareAugmentor (forward + loss + backward)")


# ── CNN baseline tests ──────────────────────────────────────────────────────

def test_cnn_classifier():
    from models.cnn_baseline import CNNClassifier
    model  = CNNClassifier()
    imgs   = _make_batch()
    logits = model(imgs)
    assert logits.shape == (4,), f"CNN output shape: {logits.shape}"
    print("  [PASS] CNNClassifier")


# ── proposed system tests ────────────────────────────────────────────────────

def test_cnn_backbone():
    from models.proposed_system import CNNBackbone
    bb   = CNNBackbone()
    imgs = _make_batch()
    feat = bb(imgs)
    assert feat.shape == (4, 256), f"Backbone output: {feat.shape}"
    print("  [PASS] CNNBackbone")


def test_prototype_head():
    from models.proposed_system import PrototypeHead
    head  = PrototypeHead(feature_dim=256, n_classes=2)
    s_f   = torch.rand(10, 256)      # 5 support per class
    s_l   = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    q_f   = torch.rand(8, 256)
    logits = head(s_f, s_l, q_f)
    assert logits.shape == (8, 2), f"Head logits shape: {logits.shape}"
    print("  [PASS] PrototypeHead")


def test_proposed_system_forward():
    from models.proposed_system import ProposedFewShotSystem
    model   = ProposedFewShotSystem()
    s_imgs  = _make_batch(b=10)
    s_labels= torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    q_imgs  = _make_batch(b=8)
    logits  = model(s_imgs, s_labels, q_imgs)
    assert logits.shape == (8, 2), f"System logits shape: {logits.shape}"
    loss = ProposedFewShotSystem.episode_loss(logits, torch.randint(0, 2, (8,)))
    loss.backward()
    print("  [PASS] ProposedFewShotSystem (forward + loss + backward)")


def test_proposed_inference():
    from models.proposed_system import ProposedFewShotSystem
    model   = ProposedFewShotSystem()
    model.eval()
    s_imgs  = _make_batch(b=10)
    s_labels= torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    model.build_support_cache(s_imgs, s_labels)
    q_imgs  = _make_batch(b=6)
    preds   = model.predict(q_imgs)
    assert preds.shape == (6,), f"Prediction shape: {preds.shape}"
    assert all(p in [0, 1] for p in preds.tolist()), "Predictions must be 0 or 1"
    print("  [PASS] ProposedFewShotSystem (inference path)")


# ── preprocessing tests ─────────────────────────────────────────────────────

def test_psnr_ssim():
    from data.preprocessing import psnr, ssim_score
    from PIL import Image
    import numpy as np

    a = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    p = psnr(a, a)
    assert p == float("inf"), "PSNR of identical images should be inf"

    p2 = psnr(a, b)
    assert p2 < float("inf"), "PSNR of different images should be finite"

    s = ssim_score(a, a)
    assert abs(s - 1.0) < 1e-4, f"SSIM of identical images should be ~1: {s}"
    print("  [PASS] PSNR and SSIM")


# ── run all ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRunning sanity tests ...\n")

    tests = [
        # augmentation
        test_image_encoder,
        test_structural_extractor,
        test_fusion_module,
        test_image_decoder,
        test_full_augmentor,
        # baseline
        test_cnn_classifier,
        # proposed system
        test_cnn_backbone,
        test_prototype_head,
        test_proposed_system_forward,
        test_proposed_inference,
        # preprocessing
        test_psnr_ssim,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
