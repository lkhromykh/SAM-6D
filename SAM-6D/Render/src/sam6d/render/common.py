import os
from importlib import resources
from pathlib import Path
from typing import Literal, NamedTuple, Self, TypeAlias

import numpy as np
import trimesh
from PIL import Image

PathLike: TypeAlias = str | os.PathLike[str] | Path
CamPoses: TypeAlias = np.ndarray[tuple[int, Literal[4], Literal[4]], np.dtype[np.floating]]


class Templates(NamedTuple):
    rgbs: np.ndarray[tuple[int, int, int, Literal[3]], np.dtype[np.uint8]]
    nocs: np.ndarray[tuple[int, int, int, Literal[3]], np.dtype[np.floating]]
    masks: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]

    def save(self, path: PathLike) -> None:
        np.savez(
            path,
            allow_pickle=False,
            rgbs=self.rgbs,
            nocs=self.nocs,
            masks=self.masks
        )

    @classmethod
    def load(cls, path: PathLike) -> Self:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                rgbs=data["rgbs"],
                nocs=data["nocs"],
                masks=data["masks"],
            )


def visualize_templates(templates: Templates, output_path: PathLike) -> None:
    path = Path(output_path).resolve()
    path.mkdir()
    rgbs = templates.rgbs
    nocs = (255 * templates.nocs).astype(np.uint8)
    masks = (255 * templates.masks).astype(np.uint8)

    for idx, (rgb, noc, mask) in enumerate(zip(rgbs, nocs, masks)):
        img_path = path / f"rgb_{idx}.jpg"
        nocs_path = path / f"nocs_{idx}.jpg"
        mask_path = path / f"mask_{idx}.jpg"
        Image.fromarray(rgb).save(img_path)
        Image.fromarray(noc).save(nocs_path)
        Image.fromarray(mask).save(mask_path)


def get_norm_info(mesh: trimesh.Trimesh, sample_points: int | None = None) -> float:
    if sample_points is None:
        min_value, max_value = mesh.bounds
    else:
        model_points, _ = trimesh.sample.sample_surface(mesh, sample_points)[0]
        min_value = np.min(model_points, axis=0)
        max_value = np.max(model_points, axis=0)
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    return 0.5 / radius


def load_poses(name: str) -> CamPoses:
    # path = resources.files(__package__).joinpath(name)
    path = Path(__file__).parent.joinpath(name)
    with path.open("rb") as file:
        return np.load(file, allow_pickle=False)
