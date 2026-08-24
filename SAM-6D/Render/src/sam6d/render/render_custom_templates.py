import blenderproc as bproc

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import trimesh

if TYPE_CHECKING:
    from sam6d.render.common import (
        CamPoses,
        PathLike,
        Templates,
        get_norm_info,
        load_poses,
    )
else:
    # todo: blender does not recognize the package
    _package = Path(__file__).parent
    sys.path.append(str(_package))
    from common import CamPoses, PathLike, Templates, get_norm_info, load_poses


def render_custom_templates(
        cad_path: PathLike,
        cam_poses: CamPoses,
        normalize: bool = True,
        color: tuple[float, float, float, float] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | None = None
) -> Templates:
    mesh = trimesh.load(str(cad_path), force='mesh')
    assert isinstance(mesh, trimesh.Trimesh)
    scale = get_norm_info(mesh)
    scale = (scale, scale, scale)
    colors = []
    nocs = []
    for cam_pose in cam_poses.copy():
        bproc.clean_up()
        obj = bproc.loader.load_obj(str(cad_path))[0]
        obj.set_cp('category_id', 1)
        if normalize:
            obj.set_scale(scale)
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
    masks = nocs[..., -1] > 0.5
    nocs = nocs[..., :3]
    return Templates(
        rgbs=colors,
        nocs=nocs,
        masks=masks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cad_path', required=True, help='The path of CAD model')
    parser.add_argument('--output_path', required=True, help='The path to save CAD templates')
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
    cam_poses = load_poses('cam_poses_level0.npy')

    bproc.init()
    templates = render_custom_templates(str(cad_path), cam_poses, args.normalize, color)
    bproc.clean_up()

    save_fpath = Path(args.output_path).resolve()
    templates.save(save_fpath)


if __name__ == '__main__':
    main()
