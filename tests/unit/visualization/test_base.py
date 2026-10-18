"""Test base visualization."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import Colormap

from torchsom.core.som import SOM
from torchsom.visualization import SOMVisualizer, VisualizationConfig

pytestmark = [
    pytest.mark.unit,
]

# Small, low-dpi figures keep rendering fast in tests.
FAST_CONFIG = VisualizationConfig(figsize=(4, 3), dpi=40)


@pytest.fixture(autouse=True)
def agg_backend() -> None:
    """Render off-screen so tests never need a display."""
    plt.switch_backend("Agg")


class TestPlotGridColormaps:
    """Colormap handling in plot_grid, shared by both topologies (issue #10)."""

    @pytest.mark.filterwarnings("error::matplotlib.MatplotlibDeprecationWarning")
    @pytest.mark.parametrize(
        "cmap",
        [None, "plasma", mpl.colormaps["magma"]],
        ids=["config-default", "name", "colormap-object"],
    )
    def test_plot_grid_accepts_name_object_or_default(
        self,
        som_small: SOM,
        tmp_path: Path,
        cmap: str | Colormap | None,
    ) -> None:
        """plot_grid renders with any colormap form and no deprecated matplotlib API.

        matplotlib 3.11 removed ``matplotlib.cm.get_cmap``; resolving a colormap
        name through it crashed every map plot.
        """
        grid = torch.rand(som_small.x, som_small.y)
        grid[0, 0] = 0.0  # masked cell, drawn with the colormap's "bad" color

        SOMVisualizer(som_small, config=FAST_CONFIG).plot_grid(
            map=grid,
            title="grid",
            colorbar_label="value",
            filename="grid",
            save_path=tmp_path,
            cmap=cmap,
        )

        assert (tmp_path / "grid.png").is_file()

    def test_plot_grid_leaves_user_colormap_unchanged(
        self,
        som_small: SOM,
        tmp_path: Path,
    ) -> None:
        """Masking NaN cells white must not mutate a colormap the caller passed in."""
        user_cmap = mpl.colormaps["magma"]
        bad_before = np.array(user_cmap.get_bad())
        grid = torch.rand(som_small.x, som_small.y)
        grid[0, 0] = 0.0

        SOMVisualizer(som_small, config=FAST_CONFIG).plot_grid(
            map=grid,
            title="grid",
            colorbar_label="value",
            filename="grid",
            save_path=tmp_path,
            cmap=user_cmap,
        )

        np.testing.assert_array_equal(np.array(user_cmap.get_bad()), bad_before)


class TestPlotAll:
    """The documented Complete Example flow (issue #11)."""

    def test_plot_all_writes_every_figure(
        self,
        som_small: SOM,
        clustered_data: tuple[torch.Tensor, torch.Tensor],
        tmp_path: Path,
    ) -> None:
        """Training then plot_all with an index BMU map saves every figure."""
        data, labels = clustered_data
        data = data.to(som_small.device)
        labels = labels.to(som_small.device)
        q_errors, t_errors = som_small.fit(data)
        bmus_map = som_small.build_map("bmus_data", data=data, return_indices=True)

        SOMVisualizer(som_small, config=FAST_CONFIG).plot_all(
            quantization_errors=q_errors,
            topographic_errors=t_errors,
            bmus_data_map=bmus_map,
            data=data,
            target=labels,
            save_path=tmp_path,
        )

        saved = {path.name for path in tmp_path.glob("*.png")}
        assert {
            "training_errors.png",
            "distance_map.png",
            "hit_map.png",
            "mean_target_map.png",
            "std_target_map.png",
            "score_map.png",
            "rank_map.png",
        } <= saved
        assert len(list((tmp_path / "component_planes").glob("*.png"))) == 4
