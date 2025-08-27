"""Clustering visualization methods for Self-Organizing Maps."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import silhouette_samples

from torchsom.core.som import SOM
from torchsom.utils.clustering import extract_clustering_features
from torchsom.visualization.config import VisualizationConfig
from torchsom.visualization.hexagonal import HexagonalVisualizer
from torchsom.visualization.rectangular import RectangularVisualizer


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
        save_path: Union[str, Path],
    ) -> Path:
        """Prepare directory for saving visualizations.

        Args:
            save_path (Union[str, Path]): Path to save the visualizations

        Returns:
            Path: Path to the saved visualizations
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _save_plot(
        self,
        save_path: Union[str, Path],
        name: str,
    ) -> None:
        """Save plot with specified configuration.

        Args:
            save_path (Union[str, Path]): Path to save the visualizations
            name (str): Name of the plot
        """
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
        n_colors: int,
        include_noise: bool = False,
    ) -> ListedColormap:
        """Create a discrete colormap for cluster visualization.

        Args:
            n_colors (int): Number of colors in the colormap
            include_noise (bool): Whether to include noise (gray)

        Returns:
            ListedColormap: Colormap with specified number of colors
        """
        effective_clusters = n_colors - 1 if include_noise else n_colors
        if effective_clusters <= 10:
            base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif effective_clusters <= 20:
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            base_colors = plt.cm.viridis(np.linspace(0, 1, effective_clusters))

        colors = base_colors[:effective_clusters]
        if include_noise:
            noise_color = np.array([[0.7, 0.7, 0.7, 1.0]])
            colors = np.vstack([noise_color, colors])

        return ListedColormap(colors)

    def plot_cluster_map(
        self,
        cluster_result: dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_values: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot clustering results overlaid on SOM grid.

        Args:
            cluster_result (dict[str, Any]): Clustering result
            title (Optional[str]): Title of the plot
            save_path (Optional[Union[str, Path]]): Path to save the plot
            show_values (bool): Whether to show values on the plot
            **kwargs: Additional arguments for the plot
        """
        method = cluster_result["method"]
        n_clusters = cluster_result["n_clusters"]
        feature_space = cluster_result.get("feature_space", "unknown")
        labels = cluster_result["labels"]

        # Reshape labels to grid
        labels_grid = labels.view(self.som.x, self.som.y)
        unique_labels = torch.unique(labels)
        has_noise = -1 in unique_labels

        # Remap labels to contiguous integers for plotting
        labels_for_plotting = labels_grid.clone()
        if has_noise:
            labels_for_plotting[labels_grid == -1] = 0
            for i, label in enumerate(unique_labels[unique_labels != -1]):
                labels_for_plotting[labels_grid == label] = i + 1
        else:
            for i, label in enumerate(unique_labels):
                labels_for_plotting[labels_grid == label] = i

        # Build discrete cmap and norm
        if labels_for_plotting.numel() > 0:
            n_bins = int(labels_for_plotting.max().item() + 1)
        else:
            n_bins = 1
        cmap = self._create_cluster_colormap(n_bins, include_noise=has_noise)
        boundaries = np.arange(-0.5, n_bins + 0.5, 1)
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)

        ticks = np.arange(n_bins)
        if has_noise:
            non_noise_labels = [
                int(v.item()) for v in unique_labels[unique_labels != -1]
            ]
            tick_labels = ["Uncertain"] + [str(v) for v in non_noise_labels]
        else:
            tick_labels = [str(int(v.item())) for v in unique_labels]

        if title is None:
            title = f"{method.upper()} Clustering ({feature_space} space)\n{n_clusters} clusters"
        if self.som.topology == "hexagonal":
            visualizer = HexagonalVisualizer(self.som, self.config)
        else:
            visualizer = RectangularVisualizer(self.som, self.config)
        visualizer.plot_grid(
            map=labels_for_plotting.float(),
            title=title,
            colorbar_label="Cluster ID",
            filename="cluster_map",
            save_path=save_path,
            cmap=cmap,
            show_values=show_values,
            value_format=".0f",
            norm=norm,
            ticks=ticks,
            tick_labels=tick_labels,
            mask_zeros=False,
            **kwargs,
        )

    def plot_silhouette_analysis(
        self,
        cluster_result: dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot silhouette analysis for clustering results.

        Args:
            cluster_result (dict[str, Any]): Clustering result
            save_path (Optional[Union[str, Path]]): Path to save the plot
        """
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
        unique_labels = np.unique(labels_clean)
        sample_silhouette_values = silhouette_samples(data_clean, labels_clean)

        fig, ax = plt.subplots(figsize=self.config.figsize)

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

        avg_score = sample_silhouette_values.mean()
        ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=2)
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

    def plot_elbow_analysis(
        self,
        max_k: int = 10,
        feature_space: str = "weights",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot elbow analysis for optimal K selection in K-means.

        Args:
            max_k (int): Maximum number of clusters to consider
            feature_space (str): Feature space to use for clustering
            save_path (Optional[Union[str, Path]]): Path to save the plot
        """
        data = extract_clustering_features(self.som, feature_space)
        n_samples = data.shape[0]
        max_k = min(max_k, n_samples - 1)
        if max_k < 2:
            print("Not enough neurons for elbow analysis")
            return

        k_range = range(2, max_k + 1)
        inertias = []
        for k in k_range:
            result = self.som.cluster(
                method="kmeans", n_clusters=k, feature_space=feature_space
            )
            inertias.append(result.get("inertia", 0))

        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax.set_title(f"Elbow Analysis for K-means ({feature_space} space)")
        ax.grid(True, alpha=0.3)
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

    def plot_cluster_quality_comparison(
        self,
        results_list: list[dict[str, Any]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Compare clustering quality metrics across different methods.

        Args:
            results_list (list[dict[str, Any]]): List of clustering results
            save_path (Optional[Union[str, Path]]): Path to save the plot
        """
        if not results_list:
            print("No clustering results to compare")
            return

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
