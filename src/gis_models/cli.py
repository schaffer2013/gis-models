from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np

from gis_models.config import BuildConfig, compute_uniform_xy_scale, route_width_projected_m
from gis_models.mesh import GridModel, build_partition_mesh
from gis_models.sources import fetch_water_polygons, load_boundary_geometry, load_route_from_kmz, write_metadata
from gis_models.terrain import (
    fetch_dem,
    fill_masked_heights,
    flatten_water,
    rasterize_geometry_mask,
    resample_array,
    union_geometries,
)


DEFAULT_KING_COUNTY_CONFIG = BuildConfig()



def _grid_axes(bounds: tuple[float, float, float, float], rows: int, cols: int, xy_scale_mm_per_m: float) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = bounds
    x = np.linspace(min_x, max_x, cols + 1)
    y = np.linspace(max_y, min_y, rows + 1)
    x0 = (min_x + max_x) / 2.0
    y0 = (min_y + max_y) / 2.0
    x_mm = (x - x0) * xy_scale_mm_per_m
    y_mm = (y - y0) * xy_scale_mm_per_m
    return x_mm, y_mm



def _corner_heights_from_cell_centers(cell_heights: np.ndarray) -> np.ndarray:
    rows, cols = cell_heights.shape
    padded = np.pad(cell_heights, 1, mode="edge")
    corners = np.zeros((rows + 1, cols + 1), dtype=float)
    for i in range(rows + 1):
        for j in range(cols + 1):
            window = padded[i : i + 2, j : j + 2]
            corners[i, j] = float(np.mean(window))
    return corners



def _export_mesh(mesh, path: Path, name: str) -> None:
    if len(mesh.faces) == 0:
        path.write_text(f"solid {name}\nendsolid {name}\n", encoding="utf-8")
        return
    mesh.export(path)



def build_model(config: BuildConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    boundary, target_crs, boundary_source = load_boundary_geometry(
        boundary_file=config.boundary_file,
        boundary_layer=config.boundary_layer,
        state_fips=config.state_fips,
        county_fips=config.county_fips,
    )
    terrain_raw_m, raw_bounds = fetch_dem(boundary, resolution_m=config.dem_resolution_m)
    terrain_m = resample_array(terrain_raw_m, (config.grid_resolution, config.grid_resolution))

    area_geom = boundary.geometry.iloc[0]
    area_mask = rasterize_geometry_mask(area_geom, raw_bounds, terrain_m.shape)
    terrain_m = fill_masked_heights(terrain_m, area_mask)

    xy_scale = compute_uniform_xy_scale(raw_bounds, config.output_width_mm, config.output_height_mm)
    z_scale = config.z_mm_per_meter

    route_mask = np.zeros_like(area_mask)
    route_feature = None
    route_width_projected = 0.0
    if config.kmz_file:
        route_feature = load_route_from_kmz(config.kmz_file, target_crs=target_crs)
        route_width_projected = route_width_projected_m(config.route_width_mm, xy_scale)
        route_buffer = route_feature.buffer(route_width_projected / 2.0, cap_style=2, join_style=2)
        route_mask = rasterize_geometry_mask(route_buffer, raw_bounds, terrain_m.shape) & area_mask

    water_mask = np.zeros_like(area_mask)
    if config.include_water:
        try:
            water_geoms = fetch_water_polygons(boundary)
            water_union = union_geometries(water_geoms)
            if not water_union.is_empty:
                water_mask = rasterize_geometry_mask(water_union, raw_bounds, terrain_m.shape) & area_mask
        except Exception as exc:  # pragma: no cover - network/runtime variability
            print(f"Warning: failed to fetch water polygons ({exc}). Continuing without water.")

    terrain_mm = terrain_m * z_scale
    terrain_mm = flatten_water(terrain_mm, water_mask, config.water_drop_mm)
    corner_z_mm = _corner_heights_from_cell_centers(terrain_mm) + config.base_thickness_mm

    route_mask &= area_mask
    terrain_only_mask = area_mask & ~route_mask

    x_mm, y_mm = _grid_axes(raw_bounds, terrain_m.shape[0], terrain_m.shape[1], xy_scale)
    terrain_mesh, terrain_stats = build_partition_mesh(
        GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=corner_z_mm, cell_mask=terrain_only_mask, base_z_mm=0.0)
    )
    route_mesh, route_stats = build_partition_mesh(
        GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=corner_z_mm, cell_mask=route_mask, base_z_mm=0.0)
    )

    terrain_path = config.output_dir / "terrain_body.stl"
    route_path = config.output_dir / "route_body.stl"
    _export_mesh(terrain_mesh, terrain_path, "terrain_body")
    _export_mesh(route_mesh, route_path, "route_body")

    metadata = {
        "build": config.to_metadata(),
        "boundary": {
            **boundary_source,
            "target_crs": str(target_crs),
            "bounds_m": {
                "min_x": raw_bounds[0],
                "min_y": raw_bounds[1],
                "max_x": raw_bounds[2],
                "max_y": raw_bounds[3],
            },
        },
        "xy_scale_mm_per_m": xy_scale,
        "route_width_projected_m": route_width_projected,
        "area_cells": int(area_mask.sum()),
        "route_cells": int(route_mask.sum()),
        "water_cells": int(water_mask.sum()),
        "terrain_mesh": asdict(terrain_stats),
        "route_mesh": asdict(route_stats),
    }
    if route_feature is not None:
        metadata["route_length_m"] = float(gpd.GeoSeries([route_feature], crs=target_crs).length.iloc[0])

    write_metadata(config.output_dir / "metadata.json", metadata)
    return metadata



def _add_common_build_arguments(parser: argparse.ArgumentParser, *, king_defaults: bool) -> None:
    parser.add_argument("--area-name", default=DEFAULT_KING_COUNTY_CONFIG.area_name if king_defaults else "Custom area")
    parser.add_argument("--boundary-file", type=Path, help="Boundary file for any area (GeoJSON, GPKG, Shapefile, etc.).")
    parser.add_argument("--boundary-layer", help="Optional layer name when reading a multi-layer boundary file.")
    parser.add_argument("--state-fips", default=DEFAULT_KING_COUNTY_CONFIG.state_fips if king_defaults else None)
    parser.add_argument("--county-fips", default=DEFAULT_KING_COUNTY_CONFIG.county_fips if king_defaults else None)
    parser.add_argument("--kmz-file", type=Path, help="Optional KMZ/KML route file to inlay into the terrain.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_KING_COUNTY_CONFIG.output_dir if king_defaults else Path("out/custom-area"))
    parser.add_argument("--output-width-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.output_width_mm)
    parser.add_argument("--output-height-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.output_height_mm)
    parser.add_argument(
        "--elevation-mm-per-1000-ft",
        type=float,
        default=DEFAULT_KING_COUNTY_CONFIG.elevation_mm_per_1000_ft,
    )
    parser.add_argument("--route-width-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.route_width_mm)
    parser.add_argument("--base-thickness-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.base_thickness_mm)
    parser.add_argument("--water-drop-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.water_drop_mm)
    parser.add_argument("--grid-resolution", type=int, default=DEFAULT_KING_COUNTY_CONFIG.grid_resolution)
    parser.add_argument("--dem-resolution-m", type=float, default=DEFAULT_KING_COUNTY_CONFIG.dem_resolution_m)
    parser.add_argument("--include-water", action=argparse.BooleanOptionalAction, default=DEFAULT_KING_COUNTY_CONFIG.include_water)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gis-models", description="Generate printable terrain models from GIS data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_area = subparsers.add_parser(
        "build-area",
        help="Build a terrain model for any area from a boundary file or a U.S. county FIPS pair.",
    )
    _add_common_build_arguments(build_area, king_defaults=False)

    build_king = subparsers.add_parser(
        "build-king-county",
        help="Build the King County, WA example model using the default 300 x 300 mm / 12 mm per 1000 ft settings.",
    )
    _add_common_build_arguments(build_king, king_defaults=True)

    return parser



def _config_from_args(args: argparse.Namespace, *, king_county_defaults: bool) -> BuildConfig:
    config = BuildConfig(
        area_name=args.area_name,
        boundary_file=args.boundary_file,
        boundary_layer=args.boundary_layer,
        state_fips=args.state_fips,
        county_fips=args.county_fips,
        output_width_mm=args.output_width_mm,
        output_height_mm=args.output_height_mm,
        elevation_mm_per_1000_ft=args.elevation_mm_per_1000_ft,
        route_width_mm=args.route_width_mm,
        base_thickness_mm=args.base_thickness_mm,
        water_drop_mm=args.water_drop_mm,
        grid_resolution=args.grid_resolution,
        dem_resolution_m=args.dem_resolution_m,
        include_water=args.include_water,
        output_dir=args.output_dir,
        kmz_file=args.kmz_file,
    )
    if king_county_defaults and config.boundary_file is None:
        config.area_name = DEFAULT_KING_COUNTY_CONFIG.area_name
        config.state_fips = DEFAULT_KING_COUNTY_CONFIG.state_fips
        config.county_fips = DEFAULT_KING_COUNTY_CONFIG.county_fips
    return config



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-area":
        metadata = build_model(_config_from_args(args, king_county_defaults=False))
    elif args.command == "build-king-county":
        metadata = build_model(_config_from_args(args, king_county_defaults=True))
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(f"Wrote outputs to {args.output_dir}")
    print(f"Route cells: {metadata['route_cells']}; water cells: {metadata['water_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
