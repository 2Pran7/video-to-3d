# run_reconstruction.py
# Minimal DUSt3R pipeline: images -> 3D point cloud (.ply). No Gradio.

import os, sys
import numpy as np
import trimesh

# Tell Python where the DUSt3R package lives (one folder down, in ./dust3r)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dust3r'))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ---- CONFIG: the only lines you'll normally edit ----
IMAGE_PATHS = ['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png']
CHECKPOINT  = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
OUTPUT_PLY  = 'reconstruction.ply'
# -----------------------------------------------------

device = 'cuda'

# 1. Load the model from your local checkpoint file
model = AsymmetricCroCo3DStereo.from_pretrained(CHECKPOINT).to(device)

# 2. Load + resize images to the resolution the model was trained on
images = load_images(IMAGE_PATHS, size=512)

# 3. Decide which images get compared against which
pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

# 4. Run the network -> raw per-pair 3D predictions
output = inference(pairs, model, device, batch_size=1)

# 5. Stitch all the pairs into one consistent global scene
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)

# 6. Pull out the pieces we need to build a point cloud
imgs  = scene.imgs          # list of HxWx3 colour images, values in 0..1
pts3d = scene.get_pts3d()   # list of HxWx3: one 3D point per pixel
masks = scene.get_masks()   # list of HxW bool: True = high-confidence point

# 7. Keep only confident points, pairing each with its pixel colour
all_points, all_colors = [], []
for img, pts, mask in zip(imgs, pts3d, masks):
    m = mask.cpu().numpy()
    all_points.append(pts.detach().cpu().numpy()[m])
    all_colors.append((img[m] * 255).astype(np.uint8))

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

# 8. Write a .ply point cloud you can open in MeshLab or an online viewer
cloud = trimesh.PointCloud(vertices=points, colors=colors)
cloud.export(OUTPUT_PLY)
print(f'Saved {len(points):,} points to {OUTPUT_PLY}')