import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from tqdm import tqdm

from stage2.transport_lq2hq import create_transport, Sampler
from utils.model_utils import instantiate_from_config
from utils.dist_utils import setup_distributed
from RAE.src import initialize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="checkpoint path (overrides config stage_2.ckpt)")
    parser.add_argument("--save_dir", type=str, default="./onestep_vis")
    parser.add_argument("--num_timesteps", type=int, default=20,
                        help="how many t points to probe (uniform in [t0,t1])")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def collect_onestep_preds(
    transport,
    ema_model_fn,
    lq_latent,      # (b, v, n, d)  — the conditioning latent
    num_timesteps,
    noiselvl,
    model_img_size,
    device,
    seed=0,
):
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )

    ts = torch.linspace(t0, t1, num_timesteps, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x0 = torch.randn(lq_latent.shape, generator=gen,
                     device=device, dtype=lq_latent.dtype)
    x0 = x0 * noiselvl + lq_latent 

    drift_fn = transport.get_drift() 

    records = []
    x_prev = x0.clone()
    t_prev = torch.zeros(lq_latent.shape[0], device=device)

    for t_scalar in tqdm(ts, desc="probing timesteps"):
        t_batch = t_scalar.expand(lq_latent.shape[0])  # (b,)

        dt = t_scalar - t_prev[0]
        if dt.abs() > 1e-6:
            v_step = drift_fn(
                x_prev, t_prev, ema_model_fn,
                model_img_size=model_img_size,
            )
            x_t = x_prev + dt * v_step
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e4, neginf=-1e4)
            x_t = torch.clamp(x_t, -1e4, 1e4)
        else:
            x_t = x_prev.clone()

        v_pred  = drift_fn(x_t, t_batch, ema_model_fn,
                           model_img_size=model_img_size)
        t_exp   = t_scalar.view(1, 1, 1, 1)         
        x1_pred = x_t - t_exp * v_pred               # x1_pred = x_t - t*v

        records.append({
            "t":       t_scalar.item(),
            "x_t":     x_t.cpu(),
            "x1_pred": x1_pred.cpu(),
        })

        x_prev = x_t.detach()
        t_prev = t_batch.detach()

    return records


def l2_norm_map(x):
    """x: (v, n, d) → per-token L2 norm → (v, n) numpy"""
    return x.norm(dim=-1).numpy()          # (v, n)


def plot_onestep_summary(records, lq_latent, hq_latent, save_dir, tag=""):
    """
    Plots:
      1. x1_pred L2 norm heatmap (v × n) for each t  → saved per-t
      2. Mean cosine sim of x1_pred vs hq_latent across t  → single curve
      3. Mean L2 distance of x1_pred vs hq_latent across t → single curve
    """
    os.makedirs(save_dir, exist_ok=True)

    hq  = hq_latent[0].float()   # (v, n, d)
    lq  = lq_latent[0].float()   # (v, n, d)

    ts         = []
    cos_sims   = []
    l2_dists   = []

    for rec in records:
        t      = rec["t"]
        x1pred = rec["x1_pred"][0].float()   # (v, n, d)

        cos = torch.nn.functional.cosine_similarity(x1pred, hq, dim=-1).mean().item()
        l2  = (x1pred - hq).norm(dim=-1).mean().item()

        ts.append(t)
        cos_sims.append(cos)
        l2_dists.append(l2)

        norm_map = l2_norm_map(x1pred)       # (v, n)
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(norm_map, aspect="auto", cmap="viridis")
        ax.set_title(f"x1_pred L2 norm  |  t={t:.3f}")
        ax.set_xlabel("token index (n)")
        ax.set_ylabel("view (v)")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{tag}norm_heatmap_t{t:.3f}.png"), dpi=80)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ts, cos_sims, "o-", color="royalblue")
    axes[0].axhline(
        torch.nn.functional.cosine_similarity(lq.reshape(-1, lq.shape[-1]),
                                               hq.reshape(-1, hq.shape[-1]), dim=-1).mean().item(),
        color="gray", linestyle="--", label="lq baseline"
    )
    axes[0].set_title("one-step x1_pred  vs  hq_latent\n(cosine similarity)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("cos sim")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ts, l2_dists, "o-", color="tomato")
    axes[1].axhline(
        (lq - hq).norm(dim=-1).mean().item(),
        color="gray", linestyle="--", label="lq baseline"
    )
    axes[1].set_title("one-step x1_pred  vs  hq_latent\n(L2 distance)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("L2")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{tag}onestep_curves.png"), dpi=120)
    plt.close(fig)
    print(f"[vis] curves saved → {save_dir}/{tag}onestep_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cam_cos, patch_cos = [], []
    cam_l2,  patch_l2  = [], []

    for rec in records:
        x1pred = rec["x1_pred"][0].float()   # (v, n+1, d)  

        cam_pred  = x1pred[:, 0:1, :]   # (v, 1, d)
        pat_pred  = x1pred[:, 1:,  :]   # (v, n, d)
        cam_hq    = hq[:, 0:1, :]
        pat_hq    = hq[:, 1:,  :]

        cam_cos.append(
            torch.nn.functional.cosine_similarity(cam_pred, cam_hq, dim=-1).mean().item()
        )
        patch_cos.append(
            torch.nn.functional.cosine_similarity(pat_pred, pat_hq, dim=-1).mean().item()
        )
        cam_l2.append((cam_pred - cam_hq).norm(dim=-1).mean().item())
        patch_l2.append((pat_pred - pat_hq).norm(dim=-1).mean().item())

    axes[0].plot(ts, cam_cos,   "o-", label="camera token", color="orange")
    axes[0].plot(ts, patch_cos, "s-", label="patch tokens",  color="steelblue")
    axes[0].set_title("cosine sim: cam vs patch tokens")
    axes[0].set_xlabel("t"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ts, cam_l2,   "o-", label="camera token", color="orange")
    axes[1].plot(ts, patch_l2, "s-", label="patch tokens",  color="steelblue")
    axes[1].set_title("L2 dist: cam vs patch tokens")
    axes[1].set_xlabel("t"); axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{tag}onestep_cam_vs_patch.png"), dpi=120)
    plt.close(fig)
    print(f"[vis] cam/patch breakdown saved → {save_dir}/{tag}onestep_cam_vs_patch.png")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    rank, world_size, device = setup_distributed()
    assert world_size == 1, "Run this script with a single GPU (no DDP needed)."

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    full_cfg = OmegaConf.load(args.config)
    IMAGENET_NORMALIZE = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    time_dist_shift = math.sqrt(
        full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base
    )

    models, processors = initialize.load_model(full_cfg, rank, device)

    ckpt_path = args.ckpt or full_cfg.stage_2.ckpt
    if ckpt_path is not None:
        initialize.load_checkpoint(
            ckpt_path,
            models['ddp_denoiser'],
            models['ema_denoiser'],
            optimizer=None,
            scheduler=None,
        )
        print(f"[ckpt] loaded from {ckpt_path}")

    models['encoder'].eval()
    models['ema_denoiser'].eval()
    ema_model_fn = models['ema_denoiser'].forward

    val_loader, _ = initialize.load_val_data(full_cfg, batch_size=1,
                                             rank=rank, world_size=world_size)
    val_batch = next(iter(val_loader))

    val_hq_views = val_batch['hq_views'].to(device)   # (1, v, 3, h, w)
    val_lq_views = val_batch['lq_views'].to(device)
    b, v, c, h, w = val_lq_views.shape

    val_hq_views = IMAGENET_NORMALIZE(
        val_hq_views.view(b*v, c, h, w)
    ).view(b, v, c, h, w)
    val_lq_views = IMAGENET_NORMALIZE(
        val_lq_views.view(b*v, c, h, w)
    ).view(b, v, c, h, w)

    feat_layer = full_cfg.mvrm.train.extract_feat_layers[0]

    with torch.no_grad():
        lq_enc_out, lq_mvrm_out = models['encoder'](
            image=val_lq_views,
            export_feat_layers=[],
            mvrm_cfg=full_cfg.mvrm.train,
            mode='train',
        )
        lq_latent = lq_mvrm_out[('extract_feat', feat_layer)]   # (b, v, n, d)

        hq_enc_out, hq_mvrm_out = models['encoder'](
            image=val_hq_views,
            export_feat_layers=[],
            mvrm_cfg=full_cfg.mvrm.train,
            mode='train',
            ref_b_idx=lq_enc_out.ref_b_idx,
        )
        hq_latent = hq_mvrm_out[('extract_feat', feat_layer)]   # (b, v, n, d)

    print(f"[data] lq_latent: {lq_latent.shape}  hq_latent: {hq_latent.shape}")

    transport = create_transport(
        **full_cfg.transport.params,
        time_dist_shift=time_dist_shift,
    )

    records = collect_onestep_preds(
        transport      = transport,
        ema_model_fn   = ema_model_fn,
        lq_latent      = lq_latent,
        num_timesteps  = args.num_timesteps,
        noiselvl       = full_cfg.mvrm.noiselvl,
        model_img_size = (h, w),
        device         = device,
        seed           = args.seed,
    )

    scene_tag = val_batch['lq_ids'][0][0] + "_"
    plot_onestep_summary(
        records    = records,
        lq_latent  = lq_latent.cpu(),
        hq_latent  = hq_latent.cpu(),
        save_dir   = args.save_dir,
        tag        = scene_tag,
    )

    torch.save(
        {
            "records":    [{k: v for k, v in r.items()} for r in records],
            "lq_latent":  lq_latent.cpu(),
            "hq_latent":  hq_latent.cpu(),
        },
        os.path.join(args.save_dir, f"{scene_tag}onestep_records.pt"),
    )
    print(f"[done] all outputs saved to {args.save_dir}")


if __name__ == "__main__":
    main()