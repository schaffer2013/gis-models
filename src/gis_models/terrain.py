from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import py3dep
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def fetch_dem(boundary_gdf: gpd.GeoDataFrame, resolution_m: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    boundary_wgs84 = boundary_gdf.to_crs("EPSG:4326")
    geom = boundary_wgs84.geometry.iloc[0]
    dem = py3dep.get_dem(geom, resolution=resolution_m, crs=str(boundary_gdf.crs))
    array = np.asarray(dem.squeeze().data, dtype=float)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2D DEM, got shape={array.shape!r}")

    x = np.asarray(dem.x, dtype=float)
    y = np.asarray(dem.y, dtype=float)
    bounds = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
    array = np.flipud(array)
    return array, bounds



def rasterize_geometry_mask(
    geometry: Polygon | MultiPolygon,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    *,
    all_touched: bool = True,
) -> np.ndarray:
    transform = from_bounds(*bounds, width=shape[1], height=shape[0])
    return features.rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)



def resample_array(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import rasterio.warp

    src_transform = from_bounds(0, 0, array.shape[1], array.shape[0], width=array.shape[1], height=array.shape[0])
    dst_transform = from_bounds(0, 0, shape[1], shape[0], width=shape[1], height=shape[0])
    dst = np.empty(shape, dtype=np.float32)
    rasterio.warp.reproject(
        source=array,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:3857",
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
    )
    return dst



def fill_masked_heights(heights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = heights.copy()
    invalid = ~np.isfinite(result) | ~valid_mask
    if not invalid.any():
        return result
    nearest_indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    result[invalid] = result[tuple(nearest_indices[:, invalid])]
    return result



def flatten_water(heights_mm: np.ndarray, water_mask: np.ndarray, drop_mm: float) -> np.ndarray:
    if not water_mask.any():
        return heights_mm
    output = heights_mm.copy()
    water_values = output[water_mask]
    reference = np.nanpercentile(water_values, 10) if water_values.size else 0.0
    output[water_mask] = reference - drop_mm
    return output



def union_geometries(geoms) -> Polygon | MultiPolygon:
    if geoms is None:
        return Polygon()
    if hasattr(geoms, "geometry"):
        geom_list = [geom for geom in geoms.geometry if geom and not geom.is_empty]
    else:
        geom_list = [geom for geom in geoms if geom and not geom.is_empty]
    if not geom_list:
        return Polygon()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return unary_union(geom_list)
