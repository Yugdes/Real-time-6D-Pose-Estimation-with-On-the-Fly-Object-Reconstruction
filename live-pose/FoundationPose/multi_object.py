# for part 3
import time
import cv2
import numpy as np
from Utils import draw_posed_3d_box, draw_xyz_axis  # or wherever they are defined
import nvdiffrast.torch as dr
# Part 2: Initialize objects for tracking
import trimesh
import numpy as np
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
# import dr  # Assuming this is the correct module for RasterizeCudaContext
import pyrealsense2 as rs
# Part 1: Object Selection
import os
import argparse
from FoundationPose.mask import create_mask_from_image



def manual_multi_object_masking_for_multiple_objects(image, object_names):
    """
    Calls create_mask_from_image() once per object using the same input image.
    Returns a dictionary of binary masks per object.
    """
    masks = {}
    for name in object_names:
        print(f"[INFO] Draw mask for object: {name}")
        mask_path = create_mask_from_image(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise RuntimeError(f"[ERROR] No mask created or mask is unreadable for object '{name}'")

        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break

        masks[name] = mask
    return masks





def select_objects(mesh_root_dir: str):
    """
    List directories that contain 'mesh_cleaned.obj', ask the user to select from them.
    Returns a list of valid object names.
    """
    valid_objects = []
    for name in os.listdir(mesh_root_dir):
        mesh_path = os.path.join(mesh_root_dir, name, 'mesh', 'mesh_biggest_component_smoothed.obj')

        if os.path.isfile(mesh_path):
            valid_objects.append(name)

    if not valid_objects:
        raise RuntimeError(f"[ERROR] No valid objects found in {mesh_root_dir}")

    print("[INFO] Available objects:")
    print(", ".join(valid_objects))

    user_input = input("Enter the names of the objects to track (comma-separated):\n")
    selected_names = [name.strip() for name in user_input.split(",")]

    selected_objects = []
    for name in selected_names:
        if name in valid_objects:
            selected_objects.append(name)
        else:
            print(f"[WARNING] '{name}' is not a valid object or missing mesh_cleaned.obj.")

    if not selected_objects:
        raise RuntimeError("[ERROR] No valid objects selected. Exiting.")

    print(f"[INFO] Tracking objects: {', '.join(selected_objects)}")
    return selected_objects







def initialize_objects(mesh_root_dir: str, input_root_dir: str, object_names):
    """
    Initialize object estimators and metadata for all selected objects.
    Returns a dictionary of object data.
    """
    object_data = {}

    for name in object_names:
        print(f"[INFO] Initializing object: {name}")

        mesh_path = os.path.join(mesh_root_dir, name, 'mesh', 'mesh_biggest_component_smoothed.obj')

        cam_K_path = os.path.join(input_root_dir, name, 'cam_K.txt')

        if not os.path.exists(mesh_path):
            print(f"[WARNING] Mesh not found for {name}, skipping.")
            continue
        if not os.path.exists(cam_K_path):
            print(f"[WARNING] cam_K.txt not found for {name}, skipping.")
            continue

        try:
            # Load mesh and intrinsics
            mesh = trimesh.load(mesh_path)
            cam_K = np.loadtxt(cam_K_path).reshape(3, 3)

            # Compute to_origin and bbox
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            # Create estimator
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()

            estimator = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx
            )

            object_data[name] = {
                'mesh': mesh,
                'cam_K': cam_K,
                'bbox': bbox,
                'to_origin': to_origin,
                'estimator': estimator,
                'initialized': False,
                'pose': None
            }

            print(f"[INFO] Loaded and initialized: {name}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize {name}: {e}")

    if not object_data:
        raise RuntimeError("[ERROR] No objects successfully initialized.")

    return object_data

# Part 3: Mask Handling & Stream Setup



def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale

# def generate_default_masks(object_names, height, width):
#     """
#     Creates dummy full-image masks (all 1s) for each object.
#     """
#     masks = {}
#     for name in object_names:
#         masks[name] = np.ones((height, width), dtype=np.uint8)
#     return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_root', type=str, default='/workspace/Object_Reconstruction/mesh',
                        help='Path to the root mesh directory containing per-object folders.')
    parser.add_argument('--input_root', type=str, default='/workspace/Object_Reconstruction/input',
                        help='Path to folder containing cam_K.txt per object.')
    
    parser.add_argument('--est_refine_iter', type=int, default=4,
                        help='Refinement iterations during pose registration')
    
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations during tracking')

    args = parser.parse_args()

    # Part 1: Object selection
    selected_objects = select_objects(args.mesh_root)

    # ✅ Part 2: Object initialization
    object_estimators = initialize_objects(args.mesh_root, args.input_root, selected_objects)

    # Part 3: Start camera stream
    pipeline, align, depth_scale = setup_realsense()
    time.sleep(3)

    # Get one frame to determine image size
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    H, W = color_image.shape[:2]

    # Let user draw masks for each object (like single-object style)
    masks = manual_multi_object_masking_for_multiple_objects(color_image, selected_objects)


    for name in masks:
        masks[name] = cv2.resize(masks[name], (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)


    # Resize all masks just in case
    # for name in masks:
    #     masks[name] = cv2.resize(masks[name], (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    
    
    Estimating = True

    try:
        i = 0
        while Estimating:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            # Get RGB-D data
            depth_image = np.asanyarray(aligned_depth_frame.get_data()) / 1e3
            color_image = np.asanyarray(color_frame.get_data())
            depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

            # Resize (if needed, usually redundant)
            color = cv2.resize(color_image, (W, H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth_image_scaled, (W, H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.1) | (depth >= np.inf)] = 0

            # Exit loop on Enter key
            if cv2.waitKey(1) == 13:
                Estimating = False
                break

            # Loop through each object
            for name, data in object_estimators.items():
                est = data['estimator']
                cam_K = data['cam_K']
                mask = masks[name]
                bbox = data['bbox']
                to_origin = data['to_origin']

                try:
                    if not data['initialized']:
                        pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                        data['initialized'] = True
                    else:
                        pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
                    data['pose'] = pose

                    center_pose = pose @ np.linalg.inv(to_origin)

                    # Visualization
                    color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    color = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)

                except Exception as e:
                    print(f"[WARNING] Tracking failed for '{name}': {e}")
                    continue

            # Show result
            cv2.imshow("Multi-Object Tracking", color[..., ::-1])
            cv2.waitKey(1)
            i += 1

    finally:
        pipeline.stop()




