import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import os


def visualize_embeddings(json_path: str, output_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    names = list(data.keys())
    embeddings = np.array(list(data.values()))

    print(f"Loaded {len(names)} embeddings of dimension {embeddings.shape[1]}")

    if len(names) < 2:
        print(
            "Warning: Need at least 2 points for PCA. Generating dummy noise points for visualization demo."
        )
        noise = np.random.normal(0, 0.1, (5, embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, embeddings + noise])
        names = names + [f"Simulated_Neighbor_{i}" for i in range(5)]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=100)

    for i, name in enumerate(names):
        short_name = name.split("/")[-1]
        plt.annotate(
            short_name,
            (pca_result[i, 0], pca_result[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Universal SANE: LLM Weight Space Embeddings (PCA)")
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="./output_eval/model_embeddings.json"
    )
    parser.add_argument(
        "--output", type=str, default="./output_eval/embedding_plot.png"
    )
    args = parser.parse_args()

    visualize_embeddings(args.input, args.output)
