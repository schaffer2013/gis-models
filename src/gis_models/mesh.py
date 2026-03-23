from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(slots=True)
class GridModel:
    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    cell_mask: np.ndarray
    base_z_mm: float
    center_z_mm: np.ndarray | None = None


@dataclass(slots=True)
class MeshStats:
    vertices: int
    faces: int
    cells: int


def _use_tl_br_diagonal(model: GridModel, i: int, j: int) -> bool:
    tl = np.array([model.x_mm[j], model.y_mm[i], model.z_mm[i, j]], dtype=float)
    tr = np.array([model.x_mm[j + 1], model.y_mm[i], model.z_mm[i, j + 1]], dtype=float)
    bl = np.array([model.x_mm[j], model.y_mm[i + 1], model.z_mm[i + 1, j]], dtype=float)
    br = np.array([model.x_mm[j + 1], model.y_mm[i + 1], model.z_mm[i + 1, j + 1]], dtype=float)
    tl_br = np.linalg.norm(br - tl)
    tr_bl = np.linalg.norm(bl - tr)
    return tl_br <= tr_bl


def build_partition_mesh(model: GridModel) -> tuple[trimesh.Trimesh, MeshStats]:
    rows, cols = model.cell_mask.shape
    if model.z_mm.shape != (rows + 1, cols + 1):
        raise ValueError("z_mm must be defined on grid corners")
    if model.center_z_mm is not None and model.center_z_mm.shape != (rows, cols):
        raise ValueError("center_z_mm must be defined on grid cells")

    top_vertex_map: dict[tuple[int, int], int] = {}
    bottom_vertex_map: dict[tuple[int, int], int] = {}
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def top_idx(i: int, j: int) -> int:
        key = (i, j)
        idx = top_vertex_map.get(key)
        if idx is None:
            idx = len(vertices)
            top_vertex_map[key] = idx
            vertices.append([float(model.x_mm[j]), float(model.y_mm[i]), float(model.z_mm[i, j])])
        return idx

    def bottom_idx(i: int, j: int) -> int:
        key = (i, j)
        idx = bottom_vertex_map.get(key)
        if idx is None:
            idx = len(vertices)
            bottom_vertex_map[key] = idx
            vertices.append([float(model.x_mm[j]), float(model.y_mm[i]), float(model.base_z_mm)])
        return idx

    cell_count = int(model.cell_mask.sum())
    if cell_count == 0:
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        return mesh, MeshStats(vertices=0, faces=0, cells=0)

    for i in range(rows):
        for j in range(cols):
            if not model.cell_mask[i, j]:
                continue
            tl = top_idx(i, j)
            tr = top_idx(i, j + 1)
            bl = top_idx(i + 1, j)
            br = top_idx(i + 1, j + 1)

            btl = bottom_idx(i, j)
            btr = bottom_idx(i, j + 1)
            bbl = bottom_idx(i + 1, j)
            bbr = bottom_idx(i + 1, j + 1)

            if model.center_z_mm is not None:
                c = len(vertices)
                vertices.append([
                    float((model.x_mm[j] + model.x_mm[j + 1]) / 2.0),
                    float((model.y_mm[i] + model.y_mm[i + 1]) / 2.0),
                    float(model.center_z_mm[i, j]),
                ])
                faces.extend([
                    [tl, bl, c],
                    [bl, br, c],
                    [br, tr, c],
                    [tr, tl, c],
                ])
            else:
                if _use_tl_br_diagonal(model, i, j):
                    faces.extend([
                        [tl, bl, br],
                        [tl, br, tr],
                    ])
                else:
                    faces.extend([
                        [tl, bl, tr],
                        [tr, bl, br],
                    ])

            faces.extend([
                [btl, bbr, bbl],
                [btl, btr, bbr],
            ])

            neighbor_specs = [
                ((i - 1, j), (tl, tr, btr, btl)),
                ((i + 1, j), (br, bl, bbl, bbr)),
                ((i, j - 1), (bl, tl, btl, bbl)),
                ((i, j + 1), (tr, br, bbr, btr)),
            ]
            for (ni, nj), (a, b, bb, ba) in neighbor_specs:
                if 0 <= ni < rows and 0 <= nj < cols and model.cell_mask[ni, nj]:
                    continue
                faces.extend([
                    [a, ba, bb],
                    [a, bb, b],
                ])

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64), process=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh, MeshStats(vertices=len(mesh.vertices), faces=len(mesh.faces), cells=cell_count)
