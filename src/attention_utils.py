import numpy as np
import torch
import matplotlib.pyplot as plt
def plot_attention_map(
    attn_collector,
    layer: int = -1,
    head: int = 0,
    batch: int = 0,
    query_positions=None,
    key_positions=None,
    reduce: str = "none",   # "none" | "mean_heads" | "mean_layers"
    vmax=None,
    title: str | None = None,
    ax=None,
    show_colorbar: bool = True,
):
    """
    Plot a single attention heatmap from your `attn_collector`.

    Expected `attn_collector`:
      - list of tensors (one per layer)
      - each tensor shape is either:
          (B, H, T, S)  [common in some PyTorch versions when batch_first=True]
          (H, B, T, S)
          (T, B, H, S)  [less common]
          (T, S) / (H, T, S) etc. (we try to infer)

    Your model uses causal self-attn, so typically T == S == seq_len.

    Args:
      layer/head/batch: which slice to visualize (ignored if reduce averages them out).
      query_positions/key_positions: optional labels for axes (list/array length T/S).
      reduce:
        - "none": plot the selected (layer, head, batch)
        - "mean_heads": average over heads for the selected layer and batch
        - "mean_layers": average over layers (and heads) for the selected batch
      vmax: optional max for colormap scaling.
      title: optional plot title.
      ax: optional matplotlib Axes to draw onto.
      show_colorbar: add colorbar if True.

    Returns:
      (fig, ax, im) where `im` is the AxesImage from imshow (useful for saving).
    """


    if not isinstance(attn_collector, (list, tuple)) or len(attn_collector) == 0:
        raise ValueError("attn_collector must be a non-empty list/tuple of attention tensors (one per layer).")

    # ---- helpers ----
    def _to_4d(w: torch.Tensor) -> torch.Tensor:
        """
        Coerce to shape (B, H, T, S) as best as possible.
        """
        if not torch.is_tensor(w):
            raise TypeError("Each element of attn_collector must be a torch.Tensor.")

        # Ensure float on CPU for plotting later (but keep torch for slicing)
        w = w.detach()

        # Common cases:
        # 4D already, but unknown dim order -> infer using equalities and typical sizes.
        if w.dim() == 4:
            # Try to detect which dim is heads: often small (<=32)
            dims = list(w.shape)

            # Candidate permutations to (B,H,T,S)
            # We'll score permutations by plausibility: H small, T and S usually equal (causal self-attn),
            # and B often equals batch size.
            perms = [
                (0, 1, 2, 3),  # B,H,T,S
                (1, 0, 2, 3),  # H,B,T,S
                (1, 2, 0, 3),  # H,T,B,S
                (2, 0, 1, 3),  # T,B,H,S
                (0, 2, 1, 3),  # B,T,H,S
                (0, 1, 3, 2),  # B,H,S,T
                (1, 0, 3, 2),  # H,B,S,T
                (2, 0, 3, 1),  # T,B,S,H
                (0, 2, 3, 1),  # B,T,S,H
            ]

            def score(p):
                B, H, T, S = [dims[i] for i in p]
                sc = 0
                if H <= 64:
                    sc += 2
                if T == S:
                    sc += 2
                if B >= 1:
                    sc += 1
                # Prefer T reasonably larger than H
                if T >= H:
                    sc += 1
                return sc

            best_p = max(perms, key=score)
            w = w.permute(*best_p).contiguous()  # -> (B,H,T,S)
            return w

        # If 3D: could be (H,T,S) or (B,T,S) etc.
        if w.dim() == 3:
            a, b, c = w.shape
            # If first dim looks like heads (small), assume (H,T,S)
            if a <= 64 and b == c:
                w = w.unsqueeze(0)  # (1,H,T,S)
                return w
            # If first dim looks like batch and b==c, assume (B,T,S) -> (B,1,T,S)
            if b == c:
                w = w.unsqueeze(1)  # (B,1,T,S)
                return w
            # Otherwise: ambiguous
            raise ValueError(f"3D attention tensor shape {tuple(w.shape)} is ambiguous; please check and adapt.")

        # If 2D: (T,S) -> (1,1,T,S)
        if w.dim() == 2:
            w = w.unsqueeze(0).unsqueeze(0)
            return w

        raise ValueError(f"Unsupported attention tensor dim={w.dim()} with shape {tuple(w.shape)}.")

    # ---- select / reduce ----
    # Convert all layers to (B,H,T,S)
    Ws = [_to_4d(w) for w in attn_collector]

    if reduce not in {"none", "mean_heads", "mean_layers"}:
        raise ValueError('reduce must be one of: "none", "mean_heads", "mean_layers".')

    if reduce == "mean_layers":
        # average across layers and heads, keep batch
        # stack: (L,B,H,T,S) -> mean over (L,H) => (B,T,S)
        W = torch.stack(Ws, dim=0).mean(dim=0).mean(dim=1)  # (B,T,S)
        A = W[batch]  # (T,S)
        used_layer = "mean_layers"
        used_head = "mean_heads"
    else:
        # pick one layer
        W = Ws[layer]  # (B,H,T,S)
        if reduce == "mean_heads":
            A = W[batch].mean(dim=0)  # (T,S)
            used_layer = layer
            used_head = "mean_heads"
        else:
            A = W[batch, head]  # (T,S)
            used_layer = layer
            used_head = head

    A = A.cpu().float().numpy()

    # ---- optionally subset positions ----
    # query_positions indexes rows (queries), key_positions indexes cols (keys)
    if query_positions is not None:
        q_idx = np.asarray(query_positions, dtype=int)
        A = A[q_idx, :]
    if key_positions is not None:
        k_idx = np.asarray(key_positions, dtype=int)
        A = A[:, k_idx]

    # ---- plot ----
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(A, aspect="auto", origin="lower", vmax=vmax)

    if title is None:
        title = f"Attention (layer={used_layer}, head={used_head}, batch={batch})"
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Label ticks if full labels were provided (not indices)
    # (If you pass arrays of indices, ticks won't be pretty; pass labels separately if you want.)
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax, im
