# gis-models

Python tooling for an STL-first terrain workflow: correlate a supplied terrain STL to its real NZ footprint, fit it into a target print volume, and split out a separate route insert.

## Current Direction

The project is now centered on this workflow:

1. Supply a terrain STL for a New Zealand landmass or region.
2. Supply the real-world boundary for that same area.
3. Supply a route file.
4. Supply print-fit parameters:
   - `max-x-mm`
   - `max-y-mm`
   - optional `max-z-mm`
   - `base-thickness-mm`
   - optional `water-distance-mm`
   - `route-thickness-mm`
   - `gap-xy-mm`
   - optional `gap-z-mm`
   - optional `route-width-mm` for line-based routes
5. The tool correlates the STL to true space, rotates it to use the build area well, then writes:
   - `terrain_base.stl`
   - `route_insert.stl`
   - `metadata.json`

## What The NZ STL Workflow Does

- Loads a user-supplied NZ boundary in `EPSG:2193` by default.
- Correlates that boundary to the supplied STL footprint with a 2D similarity fit.
- Chooses a final XY rotation that maximizes usage of the requested `max-x-mm` and `max-y-mm`.
- Respects `max-z-mm` when provided by limiting relief plus base thickness.
- Samples the correlated STL surface onto a printable grid.
- Cuts a route cavity into the base.
- Keeps the route insert at the requested thickness.
- Expands the terrain-side cavity by `gap-xy-mm`.
- Adds optional vertical clearance under the insert with `gap-z-mm`.
- Optionally extends the model outward into surrounding water by `water-distance-mm`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Primary Command

```bash
gis-models build-nz-stl-route \
  --terrain-stl /path/to/nz-area.stl \
  --boundary-file /path/to/nz-area-boundary.gpkg \
  --route-file /path/to/route.kmz \
  --output-dir out/nz-route \
  --area-name "South Island" \
  --max-x-mm 180 \
  --max-y-mm 120 \
  --max-z-mm 24 \
  --base-thickness-mm 4 \
  --route-width-mm 5 \
  --route-thickness-mm 2.5 \
  --gap-xy-mm 0.2 \
  --gap-z-mm 0.0 \
  --water-distance-mm 3 \
  --grid-resolution 1200
```

## Required Inputs

- `--terrain-stl`: the supplied terrain STL that becomes the terrain source.
- `--boundary-file`: the real-world polygon boundary for the same NZ area.
- `--output-dir`: where outputs are written.
- `--max-x-mm` and `--max-y-mm`: the available print envelope in XY.
- `--base-thickness-mm`: base thickness under the normalized terrain.
- `--route-thickness-mm`: true thickness of the route insert.

## Common Optional Inputs

- `--route-file`: route geometry in `.kmz`, `.kml`, or any vector format GeoPandas can read.
- `--route-width-mm`: printable width of a line route. Default: `5`.
- `--max-z-mm`: constrains total relief plus base thickness.
- `--gap-xy-mm`: extra clearance cut only into the terrain body. Default: `0.2`.
- `--gap-z-mm`: vertical clearance from cavity floor to route insert bottom. Default: `0.0`.
- `--water-distance-mm`: extra coastal distance beyond land. Default: `0.0`.
- `--analysis-crs`: defaults to `EPSG:2193`, which is the intended NZ workflow CRS.

## Outputs

The NZ STL route command writes:

- `terrain_base.stl`
- `route_insert.stl`
- `metadata.json`

`metadata.json` includes:

- correlation fit details
- final print-fit transform
- build-volume usage
- route and cavity cell counts
- output file locations

## Notes

- The STL is the terrain source of truth in this workflow; DEM generation is no longer the main path.
- The route split is grid-based, not a CAD boolean. Increase `--grid-resolution` if you want crisper route edges.
- The route insert keeps the requested thickness; the terrain body gets the additional XY and optional Z clearance.
- If you extend into water, the extra coastal region is inferred from the correlated terrain edge rather than fetched from a DEM.

## Legacy Commands

Older DEM-first commands are still present for now:

- `build-area`
- `build-king-county`
- `correlate-nz-reference`

They remain usable, but the NZ STL workflow above is now the intended direction for this project.
