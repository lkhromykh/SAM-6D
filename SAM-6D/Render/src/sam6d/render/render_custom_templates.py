import blenderproc as bproc

import argparse
from pathlib import Path
from typing import Literal, NamedTuple, Self, TypeAlias

import cv2
import numpy as np
import trimesh

PathLike: TypeAlias = str | Path


class Templates(NamedTuple):
    colors: np.ndarray[tuple[int, int, int, Literal[3]], np.dtype[np.uint8]]
    nocs: np.ndarray[tuple[int, int, int, Literal[3]], np.dtype[np.floating]]
    masks: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

    def save(self, dir: PathLike) -> None:
        dir = Path(dir).resolve()
        dir.mkdir(exist_ok=True, parents=True)
        gen = zip(self.colors, self.nocs.astype(np.float16), self.masks)
        for i, (color, nocs, mask) in enumerate(gen):
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dir / f'rgb_{i}.png', color)
            np.save(dir / f'xyz_{i}.npy', nocs)
            cv2.imwrite(dir / f'mask_{i}.png', mask)

    @classmethod
    def load(cls, dir: PathLike) -> Self:
        dir = Path(dir).resolve()
        if not dir.is_dir():
            raise ValueError(dir)
        colors = []
        nocs = []
        masks = []
        size = len(list(dir.glob('*.npy')))
        for i in range(size):
            color = cv2.imread(dir / f'rgb_{i}.png')
            if color is None:
                raise FileNotFoundError()
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            noc = np.load(dir / f'xyz_{i}.npy', allow_pickle=False)
            mask = cv2.imread(dir / f'mask_{i}.png', cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError()
            colors.append(color)
            nocs.append(noc)
            masks.append(mask)
        return cls(
            colors=np.stack(colors),
            nocs=np.stack(nocs),
            masks=np.stack(masks),
        )


def get_norm_info(mesh: trimesh.Trimesh) -> float:
    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)
    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    return 1/(2*radius)


def render_custom_templates(
        cad_path: PathLike,
        cam_poses: np.ndarray[tuple[int, Literal[4], Literal[4]], np.dtype[np.floating]],
        normalize: bool = True,
        color: tuple[float, float, float, float] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | None = None
) -> Templates:
    mesh = trimesh.load(cad_path, force='mesh')
    assert isinstance(mesh, trimesh.Trimesh)
    scale = get_norm_info(mesh) if normalize else 1.0
    colors = []
    nocs = []
    for cam_pose in cam_poses.copy():
        bproc.clean_up()
        obj = bproc.loader.load_obj(str(cad_path))[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp('category_id', 1)
        if color is not None:
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            obj.set_material(0, material)

        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.002
        bproc.camera.add_camera_pose(cam_pose)

        light = bproc.types.Light()
        light.set_type('POINT')
        light.set_location(2.5 * cam_pose[:3, -1])
        light.set_energy(1000)

        bproc.renderer.set_max_amount_of_samples(50)
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_nocs())
        colors.append(data['colors'][0])
        nocs.append(data['nocs'][0])

    colors = np.stack(colors)
    nocs = np.stack(nocs)
    masks = 255 * nocs[..., -1]
    nocs = 2 * nocs[..., :3] - 1
    return Templates(
        colors=colors,
        nocs=nocs,
        masks=masks.astype(np.uint8)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cad_path', required=True, help='The path of CAD model')
    parser.add_argument('--output_dir', required=True, help='The path to save CAD templates')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True, help='Whether to normalize CAD model or not')
    parser.add_argument('--colorize', action=argparse.BooleanOptionalAction, default=False, help='Whether to colorize CAD model or not')
    parser.add_argument('--base_color', default=0.05, type=float, help='The base color used in CAD model')
    args = parser.parse_args()

    cad_path = Path(args.cad_path).resolve()
    if not cad_path.exists():
        raise RuntimeError('File specified in cad_path does not exist.')

    if args.colorize:
        c = args.base_color
        color = (c, c, c, 0.0)
    else:
        color = None
    cnos_cam_fpath = Path(__file__).parent.joinpath('cam_poses_level0.npy')
    cam_poses = np.load(cnos_cam_fpath, allow_pickle=False)

    bproc.init()
    templates = render_custom_templates(str(cad_path), cam_poses, args.normalize, color)
    bproc.clean_up()

    save_fpath = Path(args.output_dir).resolve() / 'templates'
    templates.save(save_fpath)


if __name__ == '__main__':
    main()