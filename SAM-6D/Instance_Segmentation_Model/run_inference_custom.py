import torch
from PIL import Image
import logging
import os
from typing import Literal, Optional
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )


def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]
    img[edge, :] = 255

    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)

    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def save_detections(output_dir: os.PathLike, rgb: Image.Image, detections: Detections) -> None:
    detections.to_numpy()
    output_dir = os.path.join(output_dir, "sam6d_results")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "detections_ism")
    detections.save_to_file(scene_id=0, frame_id=0, runtime=0, file_path=save_path, dataset_name="Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
    save_json_bop23(save_path + ".json", detections)
    vis_ism = os.path.join(output_dir, "vis_ism.png")
    vis_img = visualize(rgb, detections, vis_ism)
    vis_img.save(vis_ism)


def run_inference(
        output_dir: os.PathLike,
        segmentor_model: Literal["sam", "fastsam"],
        cad_path: os.PathLike,
        rgb_path: os.PathLike,
        depth_path: os.PathLike,
        cam_path: os.PathLike,
        stability_score_thresh: float,
        segmentation_path: Optional[os.PathLike] = None,
) -> None:
    if segmentation_path is not None:
        run_seen(output_dir, rgb_path, segmentation_path)
    else:
        run_unseen(segmentor_model, output_dir, cad_path, rgb_path,
                   depth_path, cam_path, stability_score_thresh
                   )


def run_unseen(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    
    logging.info("Initializing template")
    template_dir = os.path.join(output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    proposal_processor = CropResizePad(target_size=224)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
        templates, token_name="x_norm_clstoken"
    ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
        templates, masks_cropped[:, 0, :, :]
    ).unsqueeze(0).data

    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    image = np.asarray(rgb)
    detections = model.segmentor_model.generate_masks(image)

    detections = Detections(detections)
    query_descriptors, query_appe_descriptors = model.descriptor_model.forward(image, detections)
    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_descriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    # compute the appearance score
    appe_scores, ref_aux_descriptor = model.compute_appearance_score(best_template, pred_idx_objects,
                                                                     query_appe_descriptors)

    # compute the geometric score
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
    )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))
    save_detections(output_dir, rgb, detections)


def run_seen(
        output_dir: os.PathLike,
        rgb_path: os.PathLike,
        segmentation_path: os.PathLike
) -> None:
    # Detections.fields = masks, object_ids, scores, boxes
    # masks shape=(N, H, W), dtype=float
    # boxes shape=(N, 4), dtype=float  [x_min, y_min, x_max, y_max]
    rgb = Image.open(rgb_path)
    masks = np.atleast_3d(np.load(segmentation_path))
    boxes = [Image.fromarray(mask).getbbox() for mask in masks]
    boxes = np.asarray(boxes)
    detections = dict(
        boxes=boxes.astype(np.float32),
        masks=masks.astype(np.float32),
        scores=np.ones(len(masks)),
        object_ids=np.zeros(len(masks))
    )
    detections = Detections(detections)
    save_detections(output_dir, rgb, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", required=True, choices=["sam", "fastsam"], help="The segmentor model in ISM")
    parser.add_argument("--output_dir", required=True, help="Path to root directory of the output")
    parser.add_argument("--cad_path", required=True, help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", required=True, help="Path to RGB image")
    parser.add_argument("--depth_path", required=True, help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", required=True, help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    parser.add_argument("--segmentation_path", help="Predefined instance mask.")
    args = parser.parse_args()
    run_inference(
        segmentor_model=args.segmentor_model,
        output_dir=args.output_dir,
        cad_path=args.cad_path,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        cam_path=args.cam_path,
        stability_score_thresh=args.stability_score_thresh,
        segmentation_path=args.segmentation_path
    )
