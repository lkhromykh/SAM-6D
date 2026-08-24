import os
from typing import Any, Literal, TypedDict

import cv2
import gorilla
import numpy as np
import torch
from sam6d.pem.model.pose_estimation_model import Net
from sam6d.pem.utils.data_utils import get_bbox, get_resize_rgb_choose
from sam6d.render.common import Templates
from torchvision.transforms import v2


class EndPointsIn(TypedDict):
    rgb: torch.Tensor
    rgb_choose: torch.Tensor
    pts: torch.Tensor
    dense_po: torch.Tensor
    dense_fo: torch.Tensor
    model: torch.Tensor


class EndPointsOut(EndPointsIn):
    pred_R:  torch.Tensor
    pred_t: torch.Tensor
    pred_pose_score: torch.Tensor


class PoseEstimationModel(torch.nn.Module):
    def __init__(
        self,
        cfg: str | os.PathLike[str],
        weights: str | os.PathLike[str],
    ) -> None:
        super().__init__()
        gcfg = gorilla.Config.fromfile(str(cfg))
        self._cfg = gcfg
        self._net = Net(gcfg.model)
        gorilla.solver.load_checkpoint(self._net, str(weights))
        self._rgb_transform = v2.Compose([
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, input_data: EndPointsIn) -> torch.Tensor:
        device = next(self.parameters()).device
        input_data = {k: torch.as_tensor(v, device=device) for k,v in input_data.items()}
        rgb = input_data['rgb']
        input_data['rgb'] = self._rgb_transform(rgb)
        output_data = self._net(input_data)
        transforms = torch.eye(4, device=device).unsqueeze(0).repeat(rgb.shape[0], 1, 1)
        transforms[:, :3, :3] = output_data['pred_R']
        transforms[:, :3, 3] = output_data['pred_t']
        transforms = torch.bmm(input_data['extrinsics'], transforms)
        return transforms

    def extract_features(self, templates: Templates) -> tuple[torch.Tensor, torch.Tensor]:
        all_rgb = []
        all_pts = []
        all_choose = []
        cfg = self._cfg.test_dataset
        for rgb, nocs, mask in zip(templates.rgbs, templates.nocs, templates.masks):
            rgb, xyz, rgb_choose = _parse_inputs(
                rgb, nocs, mask,
                cfg.img_size,
                cfg.n_sample_template_point,
                cfg.rgb_mask_flag
            )
            all_rgb.append(rgb)
            all_choose.append(rgb_choose)
            all_pts.append(xyz)

        device = next(self._net.parameters()).device
        all_rgb, all_choose, all_pts = (torch.as_tensor(np.stack(x), device=device) for x in (all_rgb, all_choose, all_pts))
        all_rgb = self._rgb_transform(all_rgb)
        all_pts = 2 * all_pts - 1  #  / 1000
        all_rgb, all_pts, all_choose = (torch.split(x, 1, 0) for x in (all_rgb, all_pts, all_choose))
        return self._net.feature_extraction.get_obj_feats(all_rgb, all_pts, all_choose)

    def extract_inputs(self, rgbs, pcds, masks):
        all_rgb = []
        all_pts = []
        all_choose = []
        cfg = self._cfg.test_dataset
        for rgb, pcd, mask in zip(rgbs, pcds, masks):
            rgb, xyz, rgb_choose = _parse_inputs(
                rgb, pcd, mask,
                cfg.img_size,
                cfg.n_sample_observed_point,
                cfg.rgb_mask_flag
            )
            all_rgb.append(rgb)
            all_pts.append(xyz)
            all_choose.append(rgb_choose)
        all_rgb, all_choose, all_pts = map(np.stack, (all_rgb, all_choose, all_pts))
        return all_rgb, all_pts, all_choose


def _to_torch(x):
    return torch.from_numpy(np.stack(x))


def _parse_inputs(
    rgb: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]],
    xyz: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.floating]],
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    img_size: int,
    n_sample_template_points: int,
    rgb_mask_flag: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    rgb = rgb[y1:y2, x1:x2]
    xyz = xyz[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    if rgb_mask_flag:
        rgb = rgb * np.expand_dims(mask, axis=-1)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # rgb = (rgb / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    # rgb = rgb.astype(np.float32)

    choose = mask.flatten().nonzero()[0]
    replace = choose.size < n_sample_template_points
    choose = np.random.choice(choose, n_sample_template_points, replace=replace)
    xyz = xyz.reshape(-1, 3)[choose]
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], img_size)
    return rgb, xyz.astype(np.float32), rgb_choose.astype(np.int64)
