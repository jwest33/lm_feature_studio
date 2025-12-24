#!/usr/bin/env python
"""Inspect and analyze saved residual activation files."""

import torch
import argparse
import sys
import json
from pathlib import Path

# Optional imports for advanced analysis
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_similarity_matrix(residuals):
    """Compute pairwise cosine similarity matrix."""
    n = residuals.shape[0]
    normalized = torch.nn.functional.normalize(residuals, dim=1)
    return torch.mm(normalized, normalized.t())


def detect_outliers_zscore(values, threshold=2.0):
    """Detect outliers using z-score method."""
    values_float = values.float()
    mean = values_float.mean()
    std = values_float.std()
    z_scores = (values_float - mean) / (std + 1e-8)
    outlier_mask = torch.abs(z_scores) > threshold
    return outlier_mask, z_scores


def detect_outliers_iqr(values, multiplier=1.5):
    """Detect outliers using IQR method."""
    values_float = values.float()
    q1 = torch.quantile(values_float, 0.25)
    q3 = torch.quantile(values_float, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    return outlier_mask, lower_bound, upper_bound


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and analyze residual activation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_residuals.py residuals.pt              # Show summary
  python inspect_residuals.py residuals.pt --list       # List all prompts
  python inspect_residuals.py residuals.pt --index 5    # Show details for index 5
  python inspect_residuals.py residuals.pt --stats      # Show tensor statistics

Analysis commands:
  python inspect_residuals.py residuals.pt --outliers              # Detect outliers
  python inspect_residuals.py residuals.pt --outliers --threshold 3.0
  python inspect_residuals.py residuals.pt --cluster 5             # K-means with 5 clusters
  python inspect_residuals.py residuals.pt --pca 3                 # PCA with 3 components
  python inspect_residuals.py residuals.pt --similarity            # Similarity analysis
  python inspect_residuals.py residuals.pt --similarity --top 10   # Top 10 similar pairs
  python inspect_residuals.py residuals.pt --dimensions            # Dimension analysis
  python inspect_residuals.py residuals.pt --dimensions --top 20   # Top 20 dimensions
  python inspect_residuals.py residuals.pt --analyze               # Run all analyses
  python inspect_residuals.py residuals.pt --export results.json   # Export to JSON
        """
    )
    parser.add_argument("file", type=str, help="Path to .pt residuals file")
    parser.add_argument("--list", "-l", action="store_true", help="List all prompts and responses")
    parser.add_argument("--index", "-i", type=int, help="Show details for specific index")
    parser.add_argument("--stats", "-s", action="store_true", help="Show tensor statistics")
    parser.add_argument("--full", "-f", action="store_true", help="Show full text (don't truncate)")

    # New analysis arguments
    parser.add_argument("--outliers", "-o", action="store_true", help="Detect outliers based on L2 norms")
    parser.add_argument("--threshold", type=float, default=2.0, help="Z-score threshold for outliers (default: 2.0)")
    parser.add_argument("--cluster", "-c", type=int, metavar="K", help="Perform k-means clustering with K clusters")
    parser.add_argument("--pca", "-p", type=int, metavar="N", help="Perform PCA with N components")
    parser.add_argument("--similarity", action="store_true", help="Compute cosine similarity analysis")
    parser.add_argument("--dimensions", "-d", action="store_true", help="Analyze activation dimensions")
    parser.add_argument("--top", type=int, default=10, help="Number of top items to show (default: 10)")
    parser.add_argument("--analyze", "-a", action="store_true", help="Run all analyses")
    parser.add_argument("--export", "-e", type=str, metavar="FILE", help="Export analysis results to JSON")
    args = parser.parse_args()

    # Load file
    try:
        data = torch.load(args.file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    residuals = data.get("residuals")
    prompts = data.get("prompts", [])
    responses = data.get("responses", [])
    indices = data.get("indices", [])

    # Summary (always shown)
    print("=" * 60)
    print("RESIDUALS FILE SUMMARY")
    print("=" * 60)
    print(f"Run ID:        {data.get('run_id', 'N/A')}")
    print(f"Timestamp:     {data.get('timestamp', 'N/A')}")
    print(f"Model:         {data.get('model_path', 'N/A')}")
    print(f"Capture point: {data.get('capture_point', 'N/A')}")
    print(f"Responses file:{data.get('responses_file', 'N/A')}")
    print(f"Num samples:   {len(prompts)}")
    if residuals is not None:
        print(f"Residual shape:{residuals.shape}")
        print(f"Dtype:         {residuals.dtype}")
    print()

    # Stats mode
    if args.stats and residuals is not None:
        print("=" * 60)
        print("TENSOR STATISTICS")
        print("=" * 60)
        print(f"Mean:          {residuals.mean().item():.6f}")
        print(f"Std:           {residuals.std().item():.6f}")
        print(f"Min:           {residuals.min().item():.6f}")
        print(f"Max:           {residuals.max().item():.6f}")
        print(f"L2 norm (avg): {residuals.norm(dim=-1).mean().item():.6f}")

        # Per-sample norms
        norms = residuals.norm(dim=-1)
        print(f"\nPer-sample L2 norms:")
        print(f"  Min:  {norms.min().item():.4f}")
        print(f"  Max:  {norms.max().item():.4f}")
        print(f"  Mean: {norms.mean().item():.4f}")
        print(f"  Std:  {norms.std().item():.4f}")
        print()

    # List mode
    if args.list:
        print("=" * 60)
        print("PROMPTS AND RESPONSES")
        print("=" * 60)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            idx = indices[i] if i < len(indices) else i
            if args.full:
                print(f"\n[{idx}] PROMPT:\n{prompt}")
                print(f"\n[{idx}] RESPONSE:\n{response}")
            else:
                prompt_preview = prompt[:100].replace('\n', ' ')
                response_preview = response[:150].replace('\n', ' ')
                print(f"[{idx}] {prompt_preview}...")
                print(f"    -> {response_preview}...")
            print("-" * 40)

    # Single index mode
    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(prompts):
            print(f"Error: Index {idx} out of range (0-{len(prompts)-1})")
            sys.exit(1)

        print("=" * 60)
        print(f"DETAILS FOR INDEX {idx}")
        print("=" * 60)
        print(f"\nPROMPT:\n{prompts[idx]}")
        print(f"\nRESPONSE:\n{responses[idx]}")

        if residuals is not None:
            r = residuals[idx]
            print(f"\nRESIDUAL:")
            print(f"  Shape: {r.shape}")
            print(f"  L2 norm: {r.norm().item():.6f}")
            print(f"  Mean: {r.mean().item():.6f}")
            print(f"  Std: {r.std().item():.6f}")
            print(f"  Top 5 values: {r.topk(5).values.tolist()}")
            print(f"  Top 5 indices: {r.topk(5).indices.tolist()}")

    # Convert residuals to float32 for analysis (bfloat16 causes issues with some ops)
    if residuals is not None and residuals.dtype != torch.float32:
        residuals = residuals.float()

    # Compute per-sample norms for export
    sample_norms = residuals.norm(dim=-1).tolist() if residuals is not None else []

    # Track results for export
    export_data = {
        "file": args.file,
        "run_id": data.get("run_id", "N/A"),
        "timestamp": data.get("timestamp", "N/A"),
        "model_path": data.get("model_path", "N/A"),
        "num_samples": len(prompts),
        "hidden_dim": residuals.shape[1] if residuals is not None else None,
        "samples": [
            {
                "index": indices[i] if i < len(indices) else i,
                "prompt": prompts[i] if i < len(prompts) else None,
                "response": responses[i] if i < len(responses) else None,
                "l2_norm": sample_norms[i] if i < len(sample_norms) else None,
            }
            for i in range(len(prompts))
        ],
        "analyses": {}
    }

    # ========================================
    # OUTLIER DETECTION
    # ========================================
    if args.outliers or args.analyze:
        if residuals is None:
            print("Error: No residuals found for outlier detection")
        else:
            print("=" * 60)
            print("OUTLIER DETECTION")
            print("=" * 60)

            norms = residuals.norm(dim=-1)

            # Z-score method
            zscore_mask, z_scores = detect_outliers_zscore(norms, args.threshold)
            zscore_outliers = torch.where(zscore_mask)[0].tolist()

            # IQR method
            iqr_mask, lower_bound, upper_bound = detect_outliers_iqr(norms)
            iqr_outliers = torch.where(iqr_mask)[0].tolist()

            print(f"\nZ-score method (threshold={args.threshold}):")
            print(f"  Found {len(zscore_outliers)} outliers")
            if zscore_outliers:
                print(f"  Outlier indices: {zscore_outliers}")
                for idx in zscore_outliers[:args.top]:
                    orig_idx = indices[idx] if idx < len(indices) else idx
                    prompt_preview = prompts[idx][:60].replace('\n', ' ') if idx < len(prompts) else "N/A"
                    print(f"    [{orig_idx}] norm={norms[idx]:.4f}, z={z_scores[idx]:.2f}: {prompt_preview}...")

            print(f"\nIQR method (1.5x IQR):")
            print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
            print(f"  Found {len(iqr_outliers)} outliers")
            if iqr_outliers:
                print(f"  Outlier indices: {iqr_outliers}")

            # Find extreme samples
            sorted_indices = torch.argsort(norms, descending=True)
            print(f"\nTop {min(5, len(norms))} highest norm samples:")
            for rank, idx in enumerate(sorted_indices[:5].tolist()):
                orig_idx = indices[idx] if idx < len(indices) else idx
                prompt_preview = prompts[idx][:50].replace('\n', ' ') if idx < len(prompts) else "N/A"
                print(f"  {rank+1}. [{orig_idx}] norm={norms[idx]:.4f}: {prompt_preview}...")

            print(f"\nTop {min(5, len(norms))} lowest norm samples:")
            for rank, idx in enumerate(sorted_indices[-5:].flip(0).tolist()):
                orig_idx = indices[idx] if idx < len(indices) else idx
                prompt_preview = prompts[idx][:50].replace('\n', ' ') if idx < len(prompts) else "N/A"
                print(f"  {rank+1}. [{orig_idx}] norm={norms[idx]:.4f}: {prompt_preview}...")

            export_data["analyses"]["outliers"] = {
                "zscore_threshold": args.threshold,
                "zscore_outliers": zscore_outliers,
                "iqr_outliers": iqr_outliers,
                "iqr_bounds": [lower_bound.item(), upper_bound.item()],
                "highest_norm_indices": sorted_indices[:5].tolist(),
                "lowest_norm_indices": sorted_indices[-5:].flip(0).tolist(),
            }
            print()

    # ========================================
    # CLUSTERING ANALYSIS
    # ========================================
    if args.cluster or args.analyze:
        if residuals is None:
            print("Error: No residuals found for clustering")
        elif not HAS_SKLEARN:
            print("Warning: sklearn not installed. Run: pip install scikit-learn")
        else:
            k = args.cluster if args.cluster else 3
            print("=" * 60)
            print(f"CLUSTERING ANALYSIS (K={k})")
            print("=" * 60)

            # Normalize for clustering
            X = residuals.numpy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Cluster statistics
            cluster_info = []
            for cluster_id in range(k):
                mask = labels == cluster_id
                cluster_indices = np.where(mask)[0]
                cluster_norms = residuals[mask].norm(dim=-1)

                info = {
                    "cluster_id": cluster_id,
                    "size": int(mask.sum()),
                    "indices": cluster_indices.tolist(),
                    "mean_norm": cluster_norms.mean().item(),
                    "std_norm": cluster_norms.std().item(),
                }
                cluster_info.append(info)

                print(f"\nCluster {cluster_id}: {info['size']} samples")
                print(f"  Mean L2 norm: {info['mean_norm']:.4f}")
                print(f"  Std L2 norm: {info['std_norm']:.4f}")
                print(f"  Sample indices: {cluster_indices[:10].tolist()}{'...' if len(cluster_indices) > 10 else ''}")

                # Show sample prompts from cluster
                for idx in cluster_indices[:3]:
                    orig_idx = indices[idx] if idx < len(indices) else idx
                    prompt_preview = prompts[idx][:60].replace('\n', ' ') if idx < len(prompts) else "N/A"
                    print(f"    [{orig_idx}] {prompt_preview}...")

            # Inertia (within-cluster sum of squares)
            print(f"\nClustering quality:")
            print(f"  Inertia: {kmeans.inertia_:.4f}")

            export_data["analyses"]["clustering"] = {
                "k": k,
                "inertia": kmeans.inertia_,
                "clusters": cluster_info,
                "labels": labels.tolist(),
            }
            print()

    # ========================================
    # PCA ANALYSIS
    # ========================================
    if args.pca or args.analyze:
        if residuals is None:
            print("Error: No residuals found for PCA")
        elif not HAS_SKLEARN:
            print("Warning: sklearn not installed. Run: pip install scikit-learn")
        else:
            n_components = args.pca if args.pca else min(10, residuals.shape[0], residuals.shape[1])
            print("=" * 60)
            print(f"PCA ANALYSIS ({n_components} components)")
            print("=" * 60)

            X = residuals.numpy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            print(f"\nExplained variance ratio:")
            cumulative = 0
            for i, var in enumerate(pca.explained_variance_ratio_):
                cumulative += var
                print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cumulative*100:.2f}%")

            print(f"\nTotal variance explained: {cumulative*100:.2f}%")

            # Show samples at extremes of PC1
            pc1 = X_pca[:, 0]
            sorted_by_pc1 = np.argsort(pc1)

            print(f"\nSamples with lowest PC1:")
            for idx in sorted_by_pc1[:3]:
                orig_idx = indices[idx] if idx < len(indices) else idx
                prompt_preview = prompts[idx][:50].replace('\n', ' ') if idx < len(prompts) else "N/A"
                print(f"  [{orig_idx}] PC1={pc1[idx]:.4f}: {prompt_preview}...")

            print(f"\nSamples with highest PC1:")
            for idx in sorted_by_pc1[-3:][::-1]:
                orig_idx = indices[idx] if idx < len(indices) else idx
                prompt_preview = prompts[idx][:50].replace('\n', ' ') if idx < len(prompts) else "N/A"
                print(f"  [{orig_idx}] PC1={pc1[idx]:.4f}: {prompt_preview}...")

            export_data["analyses"]["pca"] = {
                "n_components": n_components,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": cumulative,
                "projections": X_pca.tolist(),
            }
            print()

    # ========================================
    # SIMILARITY ANALYSIS
    # ========================================
    if args.similarity or args.analyze:
        if residuals is None:
            print("Error: No residuals found for similarity analysis")
        else:
            print("=" * 60)
            print("COSINE SIMILARITY ANALYSIS")
            print("=" * 60)

            sim_matrix = compute_similarity_matrix(residuals)
            n = sim_matrix.shape[0]

            # Get upper triangle (excluding diagonal)
            triu_indices = torch.triu_indices(n, n, offset=1)
            similarities = sim_matrix[triu_indices[0], triu_indices[1]]

            print(f"\nPairwise similarity statistics:")
            print(f"  Mean: {similarities.mean().item():.4f}")
            print(f"  Std:  {similarities.std().item():.4f}")
            print(f"  Min:  {similarities.min().item():.4f}")
            print(f"  Max:  {similarities.max().item():.4f}")

            # Find most similar pairs
            sorted_indices = torch.argsort(similarities, descending=True)
            print(f"\nTop {args.top} most similar pairs:")
            for rank in range(min(args.top, len(sorted_indices))):
                idx = sorted_indices[rank]
                i, j = triu_indices[0][idx].item(), triu_indices[1][idx].item()
                sim = similarities[idx].item()
                orig_i = indices[i] if i < len(indices) else i
                orig_j = indices[j] if j < len(indices) else j
                prompt_i = prompts[i][:100].replace('\n', ' ') if i < len(prompts) else "N/A"
                prompt_j = prompts[j][:100].replace('\n', ' ') if j < len(prompts) else "N/A"
                print(f"  {rank+1}. [{orig_i}] vs [{orig_j}]: {sim:.4f}")
                print(f"       {prompt_i}...")
                print(f"       {prompt_j}...")

            # Find most dissimilar pairs
            print(f"\nTop {args.top} most dissimilar pairs:")
            for rank in range(min(args.top, len(sorted_indices))):
                idx = sorted_indices[-(rank+1)]
                i, j = triu_indices[0][idx].item(), triu_indices[1][idx].item()
                sim = similarities[idx].item()
                orig_i = indices[i] if i < len(indices) else i
                orig_j = indices[j] if j < len(indices) else j
                prompt_i = prompts[i][:100].replace('\n', ' ') if i < len(prompts) else "N/A"
                prompt_j = prompts[j][:100].replace('\n', ' ') if j < len(prompts) else "N/A"
                print(f"  {rank+1}. [{orig_i}] vs [{orig_j}]: {sim:.4f}")
                print(f"       {prompt_i}...")
                print(f"       {prompt_j}...")

            export_data["analyses"]["similarity"] = {
                "mean": similarities.mean().item(),
                "std": similarities.std().item(),
                "min": similarities.min().item(),
                "max": similarities.max().item(),
            }
            print()

    # ========================================
    # DIMENSION ANALYSIS
    # ========================================
    if args.dimensions or args.analyze:
        if residuals is None:
            print("Error: No residuals found for dimension analysis")
        else:
            print("=" * 60)
            print("DIMENSION ANALYSIS")
            print("=" * 60)

            hidden_dim = residuals.shape[1]

            # Variance per dimension
            dim_variance = residuals.var(dim=0)
            dim_mean = residuals.mean(dim=0)
            dim_abs_mean = residuals.abs().mean(dim=0)

            # Top dimensions by variance
            sorted_by_var = torch.argsort(dim_variance, descending=True)
            print(f"\nTop {args.top} dimensions by variance:")
            for rank, dim in enumerate(sorted_by_var[:args.top].tolist()):
                print(f"  {rank+1}. Dim {dim}: var={dim_variance[dim]:.6f}, mean={dim_mean[dim]:.6f}")

            # Top dimensions by absolute mean activation
            sorted_by_abs = torch.argsort(dim_abs_mean, descending=True)
            print(f"\nTop {args.top} dimensions by absolute mean activation:")
            for rank, dim in enumerate(sorted_by_abs[:args.top].tolist()):
                print(f"  {rank+1}. Dim {dim}: |mean|={dim_abs_mean[dim]:.6f}, var={dim_variance[dim]:.6f}")

            # Dimensions with near-zero variance (potentially unused)
            low_var_threshold = dim_variance.mean() * 0.01
            low_var_dims = torch.where(dim_variance < low_var_threshold)[0]
            print(f"\nDimensions with very low variance (<1% of mean): {len(low_var_dims)}")
            if len(low_var_dims) > 0 and len(low_var_dims) <= 20:
                print(f"  Indices: {low_var_dims.tolist()}")

            # Correlation between dimensions (sample a few)
            if residuals.shape[0] >= 10:
                print(f"\nDimension correlation sample (first 5 high-variance dims):")
                top_dims = sorted_by_var[:5].tolist()
                for i, d1 in enumerate(top_dims):
                    for d2 in top_dims[i+1:]:
                        corr = torch.corrcoef(torch.stack([residuals[:, d1], residuals[:, d2]]))[0, 1]
                        if not torch.isnan(corr):
                            print(f"  Dim {d1} vs Dim {d2}: {corr.item():.4f}")

            export_data["analyses"]["dimensions"] = {
                "hidden_dim": hidden_dim,
                "top_variance_dims": sorted_by_var[:args.top].tolist(),
                "top_abs_mean_dims": sorted_by_abs[:args.top].tolist(),
                "low_variance_count": len(low_var_dims),
                "variance_stats": {
                    "mean": dim_variance.mean().item(),
                    "std": dim_variance.std().item(),
                    "min": dim_variance.min().item(),
                    "max": dim_variance.max().item(),
                }
            }
            print()

    # ========================================
    # EXPORT RESULTS
    # ========================================
    if args.export:
        export_path = Path(args.export)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"Analysis results exported to: {export_path}")


if __name__ == "__main__":
    main()
