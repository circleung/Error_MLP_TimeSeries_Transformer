# explainability_utils.py
import torch.nn as nn
import datetime as dt
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from umap import UMAP  # requires:  pip install umap-learn
from contextlib import contextmanager


# ──────────────────────────────────────────────────────────────────────────────
#  ↓ utilities that don't depend on your project code
# ──────────────────────────────────────────────────────────────────────────────
def maybe_savefig(path: str | Path | None = None, *, dpi=300):
    """Save current Matplotlib figure, auto‑naming if path is None."""
    if path is None:
        Path("figs").mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("figs") / f"fig_{stamp}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved ⇒ {path}")


def autosave(default_name):
    """Decorator that auto‑saves a plot‑producing function."""

    def decorator(f):
        def wrapper(*args, save_path=None, **kwargs):
            ret = f(*args, **kwargs)
            maybe_savefig(save_path or default_name)
            return ret

        return wrapper

    return decorator


@autosave("attention_map.png")
def plot_attention_map(attn_map, layer=0, head=0):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_map[head].cpu().numpy(), origin="lower", aspect="auto")
    ax.set_title(f"Layer {layer} · Head {head}")
    ax.set_xlabel("Key timestep")
    ax.set_ylabel("Query timestep")
    plt.colorbar(im, ax=ax, fraction=0.046)


def token_tsne_animation(
    hidden_by_layer, save_path="figs/tsne_layers.gif", *, min_dist=0.1, n_neighbors=15
):
    """
    hidden_by_layer : list[L] of Tensor[T,B,D]  (activation after each layer)
    Saves a GIF visualising how token embeddings move in 2‑D space.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    scat = ax.scatter([], [], s=8, alpha=0.8)
    ax.set_axis_off()

    def update(i):
        hs = hidden_by_layer[i].permute(1, 0, 2)  # B,T,D
        emb = UMAP(min_dist=min_dist, n_neighbors=n_neighbors).fit_transform(
            hs.reshape(-1, hs.size(-1)).cpu().numpy()
        )  # (B*T,2)

        scat.set_offsets(emb)
        # ── rescale so frame never looks empty ────────────────────────────
        margin = 0.5
        ax.set_xlim(emb[:, 0].min() - margin, emb[:, 0].max() + margin)
        ax.set_ylim(emb[:, 1].min() - margin, emb[:, 1].max() + margin)
        ax.set_title(f"Layer {i}")
        return (scat,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(hidden_by_layer), blit=True, interval=800, repeat=False
    )
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Saved ⇒ {save_path}")


@autosave("temporal_occlusion_curve.png")
def temporal_occlusion_curve(model, src, target, loss_fn=nn.MSELoss(), save_path=None):
    src = src.float()
    target = target.float()
    base_pred = model(src)
    base_loss = loss_fn(base_pred, target).item()
    occl_err = []
    for t in range(src.size(1)):
        masked = src.clone()
        masked[:, t, :] = 0.0
        masked_loss = loss_fn(model(masked), target).item()
        print(f"True Loss: {base_loss:.6f} | Masked Loss: {masked_loss:.6f}")
        occl_err.append(masked_loss - base_loss)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.stem(range(len(occl_err)), occl_err)
    ax.set_title("Temporal occlusion – ΔLoss per masked step")
    ax.set_xlabel("Masked timestep")
    ax.set_ylabel("ΔLoss")
    return occl_err


@contextmanager
def eval_no_grad(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield
    if was_training:
        model.train()


# -------- incremental-mask routine -----------------------------------------
def incremental_mask_curve(
    model: nn.Module,
    src: torch.Tensor,  # [B, T, F]
    target: torch.Tensor,  # [B, ...]
    loss_fn: nn.Module = nn.MSELoss(),
    mask_order: str = "occlusion",  # "left-to-right" | "right-to-left" | "occlusion"
    save_path: str | None = None,
):
    """
    Returns a list `losses[k]` = loss when *k* timesteps are masked.
    `mask_order="occlusion"` uses single-step ΔLoss to decide which rows
    to blank first (mask the *least* useful ones first).
    """
    device = next(model.parameters()).device
    src = src.to(device).float()
    target = target.to(device).float()
    B, T, F = src.shape

    # --- decide masking order ----------------------------------------------
    if mask_order == "left-to-right":
        order = list(range(T))
    elif mask_order == "right-to-left":
        order = list(reversed(range(T)))
    elif mask_order == "occlusion":
        # one-step occlusion ΔLoss
        deltas = []
        with eval_no_grad(model):
            base_loss = loss_fn(model(src), target).item()
            for t in range(T):
                masked = src.clone()
                masked[:, t] = 0.0
                d = loss_fn(model(masked), target).item() - base_loss
                deltas.append(d)
        order = torch.argsort(torch.tensor(deltas))  # least harmful first
    else:
        raise ValueError("mask_order must be one of the listed options")

    # --- incremental masking loop ------------------------------------------
    losses = []
    running = src.clone()
    with eval_no_grad(model):
        for k in range(T + 1):  # from 0 masked .. all masked
            loss_k = loss_fn(model(running), target).item()
            losses.append(loss_k)

            if k == T:  # all done
                break
            # mask the *next* timestep in the chosen order
            running[:, order[k]] = 0.0

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(T + 1), losses, marker="o")
    ax.set_xlabel("Number of masked timesteps (k)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Incremental masking curve – order: {mask_order}")
    ax.grid(True, ls=":")
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return losses, order


# ──────────────────────────────────────────────────────────────────────────────
#  ↓ project‑specific driver – adapt paths / config as you like
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import predict
    import utils
    from torch.utils.data import DataLoader

    # ── 1. load model --------------------------------------------------------
    selected_model = "transformer_decoder"
    model_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/src/training_logs/transformer_decoder/checkpoints/epoch=7-val_loss=0.00017323-step=663040.ckpt"
    config = utils.load_config(selected_model)
    lit_model = predict.load_weights(config, model_path)
    lit_model.cuda()
    lit_model.eval()

    # ── 2. prepare one test batch -------------------------------------------
    test_dataset = utils.get_dataset(
        config["data"]["test_data_path"],
        config["data"]["sequence_length"],
        config["data"]["prediction_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    batch = next(iter(test_loader))
    past = batch["past_values"].float().cuda()
    print(f"Shape of past values: {past.shape}")
    print(f"Shape of target values: {batch['continuous_y'].shape}")
    print(f"Shape of Input Past: {batch['past_values'].shape}")

    # ── 3. forward pass with extras -----------------------------------------
    with torch.no_grad():
        preds, attn, hidden = lit_model.backbone(
            past, return_attn=True, return_hidden=True
        )

    # ── 4. plots & animation -------------------------------------------------
    plot_attention_map(attn[0][-1], layer=0, head=0)  # auto‑saved PNG
    occl_err = temporal_occlusion_curve(
        lit_model.backbone,
        past,
        batch["difference_y_cont"].cuda(),
        save_path="occlusion_curve.png",
    )
    # print("Temporal occlusion errors:", occl_err)
    # losses, order = incremental_mask_curve(
    #     lit_model.backbone,
    #     past,
    #     target,
    #     mask_order="occlusion",  # or "left-to-right"
    #     save_path="mask_curve.png",
    # )
    # print("losses:", losses)
    # print("masking order (indices):", order.tolist())

    # Animation at last
    # token_tsne_animation(hidden, save_path="tsne_layers.gif")
