import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import time, json

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
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

def visualize_all_detections(rgb, detections, output_dir="../Data/DETECTIONS"):
    print('Visualizing all detections...')
    
    # Delete all files in the output directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1

        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])

        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
        img[edge, :] = 255

        img_pil = Image.fromarray(np.uint8(img))
        save_path = f"{output_dir}/vis_det{mask_idx + 1}.png"
        img_pil.save(save_path)

        # Reset the image to the original state for the next iteration
        img = rgb.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Convert tensors in detections to lists or numpy arrays
def convert_tensors(detections):
    for key in detections:
        if isinstance(detections[key], torch.Tensor):
            detections[key] = detections[key].cpu().numpy().tolist()  # Convert to list
    return detections

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

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
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
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
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
    start_time = time.time()
    rgb = Image.open(rgb_path).convert("RGB")
    #detections1 = model.segmentor_model.generate_masks(np.array(rgb))
    
    # Save masks to masks_numpy
    # detections1['masks'][8] = detections1['masks'][7]
    # detections1['masks'][9] = detections1['masks'][7]
    # masks_numpy = detections1['masks'][7:10].cpu().numpy()
    # np.save('../Data/MASKS/masks.npy', masks_numpy)
    
    # If you want to save as an image, you need to handle the mask appropriately
    # For example, if masks are binary, you can save each mask as an image
    # for i, mask in enumerate(masks_numpy):
        # img = Image.fromarray((mask * 255).astype(np.uint8))
        # img.save(f'../Data/MASKS/mask_{i}.png')
    
    # Load masks file
    masks_numpy = np.load('../Data/MASKS/masks.npy')   
    masks_tensor = torch.from_numpy(masks_numpy).to('cuda:0')
    
    # Load boxes data
    boxes_tensor = torch.tensor([
        [404., 320., 584., 512.],
        [404., 320., 584., 512.],
        [404., 320., 584., 512.]], device='cuda:0')
    
    detections = {}
    
    # for key in detections.keys():
        # print(key)
    
    # Create a new list for masks containing only the element at index 7
    detections['masks'] = masks_tensor

    # Create a new list for boxes containing only the element at index 7
    detections['boxes'] = boxes_tensor
    
    # for key in ['masks', 'boxes']:
        # print(f"Key: {key}")
        # print(f"Values: {detections[key]}")
        # #print(detections[key][7])
        # print()
    
    
    #print(dir(detections))
    
    detections = Detections(detections)
    
    #id_selected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device='cuda:0')
    #detections.filter(id_selected)
    
    # for i, mask in enumerate(detections.masks):
        # print(i)
        # # Move the tensor to the CPU
        # mask = mask.cpu()
        
        # # Convert the mask to a PIL image
        # mask_image = Image.fromarray((mask.numpy() * 255).astype(np.uint8))

        # # Save the image
        # save_path = f"../Data/DETECTIONS_INITIAL/detection_{i}.png"           
        # mask_image.save(save_path)
    
    print(f'detections number ----->{len(detections)}')
    print(f'detections type ----->{type(detections)}')
    
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Segmentation Time: {round(execution_time, 2)} seconds\n")
    
    print(f'query_decriptors number ----->{len(query_decriptors)}')
    print(f'query_appe_descriptors number ----->{len(query_appe_descriptors)}')

    # matching descriptors (1)
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    
    print(f'idx_selected_proposals number ----->{len(idx_selected_proposals)}')
    print(f'pred_idx_objects ----->{pred_idx_objects}')
    print(f'idx_selected_proposals ----->{idx_selected_proposals}')
    #print(f'query_appe_descriptors ----->{query_appe_descriptors}\n')

    # compute the appearance score (2)
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score (3)
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000 # !!! default = 1000
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )
    
    print(f'\nbest_template ----->{best_template}\n') # list with ids of best templates for idx_selected_proposals 
    print(f'semantic_score ----->{semantic_score}\n')
    print(f'appe_scores ----->{appe_scores}\n')
    print(f'geometric_score ----->{geometric_score}\n')
    
    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio) # default
    #final_score = geometric_score
    
    final_score[0] = 0.9 # testing !!! [7] - moouse box
    print(f'final_score ----->{final_score}\n')

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/sam6d_results/vis_ism.png")
    vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
    
    #visualize_all_detections(rgb, detections)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM") # default=0.97
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )