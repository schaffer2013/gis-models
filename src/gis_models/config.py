from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

FEET_PER_METER = 3.280839895013123


@dataclass(slots=True)
class BuildConfig:
    area_name: str = "King County, WA"
    boundary_file: Path | None = None
    boundary_layer: str | None = None
    state_fips: str | None = "53"
    county_fips: str | None = "033"
    output_width_mm: float = 300.0
    output_height_mm: float = 300.0
    elevation_mm_per_1000_ft: float = 12.0
    route_width_mm: float = 5.0
    base_thickness_mm: float = 4.0
    water_drop_mm: float = 0.8
    grid_resolution: int = 700
    include_water: bool = True
    dem_resolution_m: float = 90.0
    output_dir: Path = Path("out/king-county")
    kmz_file: Path | None = None

    @property
    def z_mm_per_meter(self) -> float:
        return self.elevation_mm_per_1000_ft * FEET_PER_METER / 1000.0

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["kmz_file"] = str(self.kmz_file) if self.kmz_file else None
        payload["boundary_file"] = str(self.boundary_file) if self.boundary_file else None
        payload["z_mm_per_meter"] = self.z_mm_per_meter
        return payload


def compute_uniform_xy_scale(bounds_m: tuple[float, float, float, float], width_mm: float, height_mm: float) -> float:
    min_x, min_y, max_x, max_y = bounds_m
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Bounds must have positive width and height")
    return min(width_mm / span_x, height_mm / span_y)


def route_width_projected_m(route_width_mm: float, xy_scale_mm_per_m: float) -> float:
    if xy_scale_mm_per_m <= 0:
        raise ValueError("xy_scale_mm_per_m must be positive")
    return route_width_mm / xy_scale_mm_per_m


def utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    if not 1 <= zone <= 60:
        raise ValueError(f"Computed invalid UTM zone {zone} for lon={lon}")
    return (32600 if lat >= 0 else 32700) + zone
