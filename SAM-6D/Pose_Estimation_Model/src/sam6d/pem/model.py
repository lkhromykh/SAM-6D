import os
from typing import Any, TypedDict, NamedTuple

import cv2
import gorilla
from PIL import Image
import torch
import torchvision.transforms as transforms

from sam6d.pem.model.pose_estimation_model import Net


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


class Templates(NamedTuple):

class PoseEstimationModel(torch.nn.Module):
    def __init__(
        self,
        cfg: str | os.PathLike[str],
        weights: str | os.PathLike[str],
    ) -> None:
        super().__init__()
        gcfg = gorilla.Config.fromfile(str(cfg))
        self._net = Net(gcfg.model)
        gorilla.solver.load_checkpoint(self._net, str(weights))
        self._templates = None
        self.register_buffer("tempaltes", _templates)

    def forward(self, image, depth, point_cloud, template)

    def forward(self, end_points: EndPointsIn) -> EndPointsOut:
        return self._net(end_points)