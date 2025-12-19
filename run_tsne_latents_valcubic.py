#!/usr/bin/env python3
"""Run t-SNE on latent vectors stored under `latents/` with prefix `valcubic`.

Usage examples:
  python run_tsne_latents_valcubic.py
  python run_tsne_latents_valcubic.py --latents-dir latents --prefix valcubic --out-dir outputs/tsne

Outputs:
  - `out_dir/tsne_{prefix}.npy` : 2D t-SNE embedding (N,2)
  - `out_dir/tsne_{prefix}.png` : scatter plot (colors by label if available)

Requires: numpy, matplotlib, scikit-learn, seaborn (optional)
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def find_latents_file(latents_dir, prefix):
    # Try common filenames and fallback to folder of .npy files
    candidates = [
        os.path.join(latents_dir, f'latents_{prefix}.npy'),
        os.path.join(latents_dir, f'{prefix}.npy'),
        os.path.join(latents_dir, prefix),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # if prefix is a directory inside latents_dir, try stacking all npy files
    dirpath = os.path.join(latents_dir, prefix)
    if os.path.isdir(dirpath):
        files = sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.npy')])
        if files:
            return files  # return list -> handled specially
    return None


def load_latents(latents_path_or_list):
    if isinstance(latents_path_or_list, list):
        arrays = [np.load(p) for p in latents_path_or_list]
        X = np.concatenate(arrays, axis=0)
        return X
    else:
        return np.load(latents_path_or_list)


def load_labels(latents_dir, prefix):
    # try common label filenames
    candidates = [
        os.path.join(latents_dir, f'labels_{prefix}.npy'),
        os.path.join(latents_dir, f'labels_{prefix}.csv'),
        os.path.join(latents_dir, f'labels_{prefix}.txt'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            if p.endswith('.npy'):
                return np.load(p)
            else:
                try:
                    import pandas as pd
                    return pd.read_csv(p, header=None).iloc[:, 0].values
                except Exception:
                    # fallback to simple text read
                    return np.loadtxt(p, dtype=int)
    return None


def ensure_2d(X):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents-dir', default='latents', help='Directory that contains latent files')
    p.add_argument('--prefix', default='valcubic', help='Prefix or filename for latents (e.g. valcubic)')
    p.add_argument('--out-dir', default='outputs/tsne', help='Output directory for embeddings and plots')
    p.add_argument('--pca-dim', type=int, default=50, help='PCA dims before t-SNE (set 0 to skip)')
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--n-iter', type=int, default=1000)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--no-plot', action='store_true', help='Do not save PNG plot (only save embeddings)')
    args = p.parse_args()

    latents_path = find_latents_file(args.latents_dir, args.prefix)
    if latents_path is None:
        print(f"ERROR: could not find latents for prefix '{args.prefix}' in '{args.latents_dir}'")
        sys.exit(2)

    print(f"Loading latents from: {latents_path}")
    X = load_latents(latents_path)
    X = ensure_2d(X)
    print(f"Latents shape: {X.shape}")

    labels = load_labels(args.latents_dir, args.prefix)
    if labels is not None:
        print(f"Loaded labels with shape: {getattr(labels, 'shape', None)}")
        # Ensure labels are a 1D numpy array
        labels = np.asarray(labels).reshape(-1)
        # Map numeric labels to requested names
        label_map = {0: 'FR-I', 1: 'FR-II', 2: 'FR-X', 3: 'R'}
        try:
            # Create mapped labels array of same length
            mapped_labels = np.array([label_map.get(int(l), str(l)) for l in labels])
        except Exception:
            mapped_labels = np.array([label_map.get(l, str(l)) for l in labels])
    else:
        mapped_labels = None

    n_samples, n_features = X.shape

    # PCA reduction if requested and helpful
    if args.pca_dim and n_features > args.pca_dim:
        print(f"Applying PCA: {n_features} -> {args.pca_dim}")
        pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
        X_reduced = pca.fit_transform(X)
        print(f"PCA done. Explained variance (sum): {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_reduced = X

    # adjust perplexity for small datasets
    perplexity = args.perplexity
    if n_samples <= 3 * perplexity:
        perplexity = max(5, n_samples // 3)
    print(f"Running t-SNE (n_samples={n_samples}, perplexity={perplexity}, n_iter={args.n_iter})")

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=args.n_iter, init='pca', random_state=args.random_state, verbose=1)
    X_tsne = tsne.fit_transform(X_reduced)
    print(f"t-SNE result shape: {X_tsne.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_npy = os.path.join(args.out_dir, f'tsne_{args.prefix}.npy')
    np.save(out_npy, X_tsne)
    print(f"Saved embedding to: {out_npy}")

    # Save CSV with coordinates and (mapped) labels
    out_csv = os.path.join(args.out_dir, f'tsne_{args.prefix}.csv')
    try:
        import pandas as pd
        df = pd.DataFrame(X_tsne, columns=['tsne_1', 'tsne_2'])
        if labels is not None:
            df['label'] = labels
            df['label_name'] = mapped_labels
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV to: {out_csv}")
    except Exception:
        # fallback to numpy savetxt
        if labels is not None and mapped_labels is not None:
            header = 'tsne_1,tsne_2,label,label_name'
            data = np.column_stack([X_tsne, labels, mapped_labels])
        else:
            header = 'tsne_1,tsne_2'
            data = X_tsne
        np.savetxt(out_csv, data, delimiter=',', header=header, comments='', fmt='%s')
        print(f"Saved CSV (fallback) to: {out_csv}")

    if not args.no_plot:
        try:
            import seaborn as sns
            sns.set(style='whitegrid')
            palette = None
        except Exception:
            palette = None

        fig, ax = plt.subplots(figsize=(7, 6))
        if mapped_labels is None:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.8)
        else:
            # color by mapped label names
            try:
                unique = np.unique(mapped_labels)
                for u in unique:
                    mask = mapped_labels == u
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=10, alpha=0.8, label=str(u))
                ax.legend(markerscale=2, fontsize='small', loc='best')
            except Exception:
                ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.8)

        ax.set_xlabel('tsne_1')
        ax.set_ylabel('tsne_2')
        ax.set_title('t-SNE embedding of Latent Vectors in the Validation Dataset')
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, f'tsne_{args.prefix}.png')
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved plot to: {out_png}")

    print('Done.')


if __name__ == '__main__':
    main()
