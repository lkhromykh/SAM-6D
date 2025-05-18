# Exporting variables
export CAMERA_PATH=$PWD/Data/Inference/camera.json
export OUTPUT_DIR=$PWD/Data/Inference/outputs
export DEPTH_PATH=$PWD/Data/Inference/depth.png
export RGB_PATH=$PWD/Data/Inference/rgb.png
#export CAD_PATH=$PWD/Data/Inference/cad_models/cube_50mm.ply

#export CAD_PATH=$PWD/Data/Inference/cad_models/fixing_nut_0000.ply
#export CAD_PATH=$PWD/Data/Inference/cad_models/angle_bracket_plastic_0001.ply
#export CAD_PATH=$PWD/Data/Inference/cad_models/flange_bushing_0002.ply
#export CAD_PATH=$PWD/Data/Inference/cad_models/rectangular_bracket_0003.ply
export CAD_PATH=$PWD/Data/Inference/cad_models/tapered_holder_0004.ply
#export CAD_PATH=$PWD/Data/Inference/cad_models/latch_0005.ply
#export CAD_PATH=$PWD/Data/Inference/cad_models/bushing_0006.ply

# Render CAD templates
cd Render &&
time blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH && #--colorize True 

# Run instance segmentation model - sam or fastsam
export SEGMENTOR_MODEL=sam

echo "------Detection & Segmentation-------"
cd ../ &&
python detection_segmentation.py --object_class 4 # 0 means fixing-nut object, 7 cube, 1 angle_bracket_plastic

echo "------Best template detection--------"
cd ./Instance_Segmentation_Model &&
time python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH &&

# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

echo "----------Pose estimation------------"
cd ../Pose_Estimation_Model &&
time python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

