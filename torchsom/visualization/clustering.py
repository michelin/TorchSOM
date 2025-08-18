"""Clustering visualization methods for Self-Organizing Maps."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_samples

from torchsom.core.som import SOM
from torchsom.visualization.config import VisualizationConfig


class ClusteringVisualizer:
    """Specialized visualizer for SOM clustering results."""

    def __init__(
        self,
        som: SOM,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Initialize the clustering visualizer.

        Args:
            som (SOM): Trained SOM instance
            config (Optional[VisualizationConfig]): Visualization configuration
        """
        self.som = som
        self.config = config or VisualizationConfig()

    def _prepare_save_path(
        self,
        save_path: Union[
            str,
            Path,
        ],
    ) -> Path:
        """Prepare directory for saving visualizations."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _save_plot(
        self,
        save_path: Union[str, Path],
        name: str,
    ) -> None:
        """Save plot with specified configuration."""
        save_path = self._prepare_save_path(save_path=save_path)
        plt.savefig(
            save_path / f"{name}.{self.config.save_format}",
            dpi=self.config.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=True,
        )
        plt.close()

    def _create_cluster_colormap(
        self,
        n_clusters: int,
    ) -> ListedColormap:
        """Create a discrete colormap for cluster visualization."""
        if n_clusters <= 10:
            base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif n_clusters <= 20:
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            base_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

        colors = base_colors[:n_clusters]
        # Add gray for noise points if needed
        colors = np.vstack([[[0.7, 0.7, 0.7, 1.0]], colors])  # Gray for noise (-1)

        return ListedColormap(colors)

    def plot_cluster_map(
        self,
        cluster_result: dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_values: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot clustering results overlaid on SOM grid."""
        labels = cluster_result["labels"]
        method = cluster_result["method"]
        n_clusters = cluster_result["n_clusters"]
        feature_space = cluster_result.get("feature_space", "unknown")

        # Reshape labels to grid
        labels_grid = labels.view(self.som.x, self.som.y)

        # Create custom colormap for clusters
        unique_labels = torch.unique(labels)
        n_unique = len(unique_labels)
        cmap = self._create_cluster_colormap(n_unique)

        # Adjust labels for colormap (shift by 1 to account for noise at index 0)
        labels_for_plotting = labels_grid.clone()
        if -1 in unique_labels:
            labels_for_plotting[labels_grid == -1] = 0
            for i, label in enumerate(unique_labels[unique_labels != -1]):
                labels_for_plotting[labels_grid == label] = i + 1
        else:
            for i, label in enumerate(unique_labels):
                labels_for_plotting[labels_grid == label] = i

        # Generate title
        if title is None:
            title = f"{method.upper()} Clustering ({feature_space} space)\n{n_clusters} clusters"

        # Import visualization modules to avoid circular imports
        if self.som.topology == "hexagonal":
            from torchsom.visualization.hexagonal import HexagonalVisualizer

            visualizer = HexagonalVisualizer(self.som, self.config)
        else:
            from torchsom.visualization.rectangular import RectangularVisualizer

            visualizer = RectangularVisualizer(self.som, self.config)

        # Plot using appropriate visualizer
        visualizer.plot_grid(
            map=labels_for_plotting.float(),
            title=title,
            colorbar_label="Cluster ID",
            filename="cluster_map",
            save_path=save_path,
            cmap=cmap,
            show_values=show_values,
            value_format=".0f",
            **kwargs,
        )

    def plot_silhouette_analysis(
        self,
        cluster_result: dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot silhouette analysis for clustering results."""
        data = cluster_result["original_data"]
        labels = cluster_result["labels"]
        method = cluster_result["method"]
        feature_space = cluster_result.get("feature_space", "unknown")

        # Convert to numpy for sklearn
        data_np = data.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Remove noise points
        noise_mask = labels_np == -1
        if noise_mask.sum() == len(labels_np):
            print("Cannot plot silhouette analysis: all points are noise")
            return

        if noise_mask.sum() > 0:
            data_clean = data_np[~noise_mask]
            labels_clean = labels_np[~noise_mask]
        else:
            data_clean = data_np
            labels_clean = labels_np

        if len(np.unique(labels_clean)) <= 1:
            print("Cannot plot silhouette analysis: only one cluster")
            return

        # Calculate silhouette scores
        sample_silhouette_values = silhouette_samples(data_clean, labels_clean)

        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Sort by cluster labels and plot
        unique_labels = np.unique(labels_clean)
        y_lower = 10

        for label in sorted(unique_labels):
            cluster_silhouette_values = sample_silhouette_values[labels_clean == label]
            cluster_silhouette_values.sort()

            size_cluster = len(cluster_silhouette_values)
            y_upper = y_lower + size_cluster

            color = plt.cm.viridis(label / len(unique_labels))
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
            y_lower = y_upper + 10

        # Add average silhouette score line
        avg_score = sample_silhouette_values.mean()
        ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=2)

        # Formatting
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.set_title(
            f"Silhouette Analysis ({method.upper()}, {feature_space} space)\n"
            f"Average Score: {avg_score:.3f}"
        )

        if save_path:
            self._save_plot(save_path, "silhouette_analysis")
        else:
            plt.show()

    def plot_cluster_quality_comparison(
        self,
        results_list: list[dict[str, Any]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Compare clustering quality metrics across different methods."""
        if not results_list:
            print("No clustering results to compare")
            return

        # Extract metrics from results
        methods = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        n_clusters_list = []

        for result in results_list:
            method = result["method"]
            feature_space = result.get("feature_space", "unknown")
            metrics = result.get("metrics", {})

            methods.append(f"{method}_{feature_space}")
            silhouette_scores.append(metrics.get("silhouette_score", 0))
            davies_bouldin_scores.append(metrics.get("davies_bouldin_score", 0))
            calinski_harabasz_scores.append(metrics.get("calinski_harabasz_score", 0))
            n_clusters_list.append(result.get("n_clusters", 0))

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        x_pos = np.arange(len(methods))

        # Silhouette Score (higher is better)
        bars1 = ax1.bar(x_pos, silhouette_scores, alpha=0.7)
        ax1.set_title("Silhouette Score (Higher = Better)")
        ax1.set_ylabel("Score")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)

        for bar, score in zip(bars1, silhouette_scores):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Davies-Bouldin Score (lower is better)
        bars2 = ax2.bar(x_pos, davies_bouldin_scores, alpha=0.7, color="orange")
        ax2.set_title("Davies-Bouldin Score (Lower = Better)")
        ax2.set_ylabel("Score")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        for bar, score in zip(bars2, davies_bouldin_scores):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Calinski-Harabasz Score (higher is better)
        bars3 = ax3.bar(x_pos, calinski_harabasz_scores, alpha=0.7, color="green")
        ax3.set_title("Calinski-Harabasz Score (Higher = Better)")
        ax3.set_ylabel("Score")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(methods, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3)

        for bar, score in zip(bars3, calinski_harabasz_scores):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Number of clusters
        bars4 = ax4.bar(x_pos, n_clusters_list, alpha=0.7, color="red")
        ax4.set_title("Number of Clusters")
        ax4.set_ylabel("Count")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(methods, rotation=45, ha="right")
        ax4.grid(True, alpha=0.3)

        for bar, count in zip(bars4, n_clusters_list):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            self._save_plot(save_path, "clustering_metrics_comparison")
        else:
            plt.show()

    def plot_clustering_comparison_grid(
        self,
        methods: list[str] = None,
        feature_spaces: list[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """Plot a grid comparing different clustering methods and feature spaces."""
        if methods is None:
            methods = ["kmeans", "gmm", "hdbscan"]
        if feature_spaces is None:
            feature_spaces = ["weights", "positions", "combined"]

        n_methods = len(methods)
        n_spaces = len(feature_spaces)

        fig, axes = plt.subplots(
            n_methods, n_spaces, figsize=(5 * n_spaces, 4 * n_methods)
        )
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        if n_spaces == 1:
            axes = axes.reshape(-1, 1)

        for i, method in enumerate(methods):
            for j, feature_space in enumerate(feature_spaces):
                ax = (
                    axes[i, j]
                    if n_methods > 1 and n_spaces > 1
                    else axes[j] if n_methods == 1 else axes[i]
                )

                try:
                    result = self.som.cluster(
                        method=method, feature_space=feature_space, **kwargs
                    )

                    labels = result["labels"].view(self.som.x, self.som.y)
                    n_clusters = result["n_clusters"]

                    unique_labels = torch.unique(labels)
                    cmap = self._create_cluster_colormap(len(unique_labels))

                    ax.imshow(labels.cpu().numpy(), cmap=cmap, aspect="auto")
                    ax.set_title(
                        f"{method.upper()}\n{feature_space}\n({n_clusters} clusters)"
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error:\n{str(e)}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{method.upper()}\n{feature_space}\nFailed")
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            self._save_plot(save_path, "clustering_methods_comparison")
        else:
            plt.show()

    def plot_elbow_analysis(
        self,
        max_k: int = 10,
        feature_space: str = "weights",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot elbow analysis for optimal K selection in K-means."""
        data = self.som._extract_clustering_features(feature_space)
        n_samples = data.shape[0]
        max_k = min(max_k, n_samples - 1)

        if max_k < 2:
            print("Not enough neurons for elbow analysis")
            return

        k_range = range(2, max_k + 1)
        inertias = []

        # Run K-means for different K values
        for k in k_range:
            result = self.som.cluster(
                method="kmeans", n_clusters=k, feature_space=feature_space
            )
            inertias.append(result.get("inertia", 0))

        # Create elbow plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        ax.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax.set_title(f"Elbow Analysis for K-means ({feature_space} space)")
        ax.grid(True, alpha=0.3)

        # Annotate points
        for k, inertia in zip(k_range, inertias):
            ax.annotate(
                f"k={k}\n{inertia:.2f}",
                (k, inertia),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        if save_path:
            self._save_plot(save_path, "elbow_analysis")
        else:
            plt.show()
