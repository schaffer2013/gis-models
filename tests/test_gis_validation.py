from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import signal

from gis_models.cli import (
    DEFAULT_KING_COUNTY_CONFIG,
    _smooth_low_gradient_heights,
    _suppress_high_elevation_spikes,
)
from gis_models.sources import load_boundary_geometry
from gis_models.terrain import fetch_dem, fill_masked_heights, rasterize_geometry_mask, resample_array


ROI_SIZE_CELLS = 10
ROI_SAMPLE_COUNT = 500
RNG_SEED = 12345
MIN_SOURCE_SAMPLES_PER_ROI = 25
MAX_MEAN_ERROR_M = 9.5
MAX_AVERAGE_MEAN_ERROR_M = 3.0
MAX_P95_VARIANCE_REL_ERROR = 0.5
MAX_AVERAGE_VARIANCE_REL_ERROR = 0.2


@pytest.mark.gis_validation
@pytest.mark.skipif(
    os.getenv("RUN_GIS_VALIDATION") != "1",
    reason="Set RUN_GIS_VALIDATION=1 to run GIS ROI validation against source DEM data.",
)
def test_king_county_roi_statistics_match_source_dem() -> None:
    config = DEFAULT_KING_COUNTY_CONFIG
    boundary, _, _ = load_boundary_geometry(state_fips=config.state_fips, county_fips=config.county_fips)

    raw_dem_m, bounds, dem_crs = fetch_dem(boundary, resolution_m=config.dem_resolution_m)
    area_geom = boundary.to_crs(dem_crs).geometry.iloc[0]

    raw_mask = rasterize_geometry_mask(area_geom, bounds, raw_dem_m.shape)
    source_dem_m = raw_dem_m.copy()
    source_valid_mask = raw_mask & np.isfinite(source_dem_m)
    model_source_dem_m = fill_masked_heights(source_dem_m, raw_mask)

    model_dem_m = resample_array(
        model_source_dem_m,
        bounds,
        dem_crs,
        (config.grid_resolution, config.grid_resolution),
    )
    model_mask = rasterize_geometry_mask(area_geom, bounds, model_dem_m.shape)
    model_dem_m = fill_masked_heights(model_dem_m, model_mask)
    model_dem_m = _smooth_low_gradient_heights(model_dem_m, model_mask)
    model_dem_m = _suppress_high_elevation_spikes(model_dem_m, model_mask)

    valid_starts = signal.convolve2d(
        model_mask.astype(np.int16),
        np.ones((ROI_SIZE_CELLS, ROI_SIZE_CELLS), dtype=np.int16),
        mode="valid",
    ) == ROI_SIZE_CELLS * ROI_SIZE_CELLS
    start_positions = np.argwhere(valid_starts)
    assert len(start_positions) >= ROI_SAMPLE_COUNT, "Not enough fully valid ROI start positions for validation."

    rng = np.random.default_rng(RNG_SEED)
    shuffled_starts = start_positions[rng.permutation(len(start_positions))]

    raw_rows, raw_cols = raw_dem_m.shape
    x_edges = np.linspace(bounds[0], bounds[2], config.grid_resolution + 1)
    y_edges = np.linspace(bounds[3], bounds[1], config.grid_resolution + 1)
    raw_x_centers = np.linspace(
        bounds[0] + (bounds[2] - bounds[0]) / (2 * raw_cols),
        bounds[2] - (bounds[2] - bounds[0]) / (2 * raw_cols),
        raw_cols,
    )
    raw_y_centers = np.linspace(
        bounds[3] - (bounds[3] - bounds[1]) / (2 * raw_rows),
        bounds[1] + (bounds[3] - bounds[1]) / (2 * raw_rows),
        raw_rows,
    )

    mean_errors_m: list[float] = []
    variance_relative_errors: list[float] = []

    for i, j in shuffled_starts:
        x0 = x_edges[j]
        x1 = x_edges[j + ROI_SIZE_CELLS]
        y_top = y_edges[i]
        y_bottom = y_edges[i + ROI_SIZE_CELLS]

        raw_x_selection = (raw_x_centers >= x0) & (raw_x_centers < x1)
        raw_y_selection = (raw_y_centers <= y_top) & (raw_y_centers > y_bottom)
        source_roi = source_dem_m[np.ix_(raw_y_selection, raw_x_selection)]
        source_roi_mask = source_valid_mask[np.ix_(raw_y_selection, raw_x_selection)]
        source_values = source_roi[source_roi_mask]
        if source_values.size < MIN_SOURCE_SAMPLES_PER_ROI:
            continue

        model_values = model_dem_m[i : i + ROI_SIZE_CELLS, j : j + ROI_SIZE_CELLS].ravel()
        mean_errors_m.append(abs(float(model_values.mean()) - float(source_values.mean())))

        source_variance = float(source_values.var())
        model_variance = float(model_values.var())
        variance_relative_errors.append(abs(model_variance - source_variance) / max(source_variance, 1.0))

        if len(mean_errors_m) == ROI_SAMPLE_COUNT:
            break

    assert len(mean_errors_m) == ROI_SAMPLE_COUNT, (
        f"Only collected {len(mean_errors_m)} ROIs with at least {MIN_SOURCE_SAMPLES_PER_ROI} source samples; "
        f"expected {ROI_SAMPLE_COUNT}."
    )

    mean_errors_m = np.asarray(mean_errors_m)
    variance_relative_errors = np.asarray(variance_relative_errors)

    assert float(mean_errors_m.mean()) <= MAX_AVERAGE_MEAN_ERROR_M, (
        f"Average ROI mean elevation error was {mean_errors_m.mean():.2f} m; "
        f"expected <= {MAX_AVERAGE_MEAN_ERROR_M:.2f} m."
    )
    assert float(np.percentile(mean_errors_m, 95)) <= MAX_MEAN_ERROR_M, (
        f"ROI mean elevation error p95 was {np.percentile(mean_errors_m, 95):.2f} m; "
        f"expected <= {MAX_MEAN_ERROR_M:.2f} m."
    )
    assert float(variance_relative_errors.mean()) <= MAX_AVERAGE_VARIANCE_REL_ERROR, (
        f"Average ROI variance relative error was {variance_relative_errors.mean():.3f}; "
        f"expected <= {MAX_AVERAGE_VARIANCE_REL_ERROR:.3f}."
    )
    assert float(np.percentile(variance_relative_errors, 95)) <= MAX_P95_VARIANCE_REL_ERROR, (
        f"ROI variance relative error p95 was {np.percentile(variance_relative_errors, 95):.3f}; "
        f"expected <= {MAX_P95_VARIANCE_REL_ERROR:.3f}."
    )
