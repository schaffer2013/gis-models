from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from gis_models.config import BuildConfig
from gis_models.terrain import fetch_dem


def test_build_config_serializes_dem_file(tmp_path) -> None:
    dem_file = tmp_path / "sample-dem.tif"
    config = BuildConfig(dem_file=dem_file)

    metadata = config.to_metadata()

    assert metadata["dem_file"] == str(dem_file)


def test_fetch_dem_uses_local_raster_and_clips_to_boundary(tmp_path) -> None:
    dem_path = tmp_path / "dem.tif"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    transform = from_origin(0.0, 4.0, 1.0, 1.0)

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)

    boundary = gpd.GeoDataFrame(geometry=[box(1.0, 1.0, 3.0, 3.0)], crs="EPSG:4326")

    clipped, bounds, dem_crs = fetch_dem(boundary, resolution_m=90.0, dem_file=dem_path)

    assert clipped.shape == (2, 2)
    assert np.allclose(clipped, np.array([[5.0, 6.0], [9.0, 10.0]]))
    assert bounds == (1.0, 1.0, 3.0, 3.0)
    assert dem_crs.to_epsg() == 4326
