# reconstruct.py
# One-command video -> 3D point cloud using DUSt3R.
# Usage:  python reconstruct.py my_video.mp4

import os, sys, glob
import numpy as np
import cv2
import torch
import trimesh

# Make the DUSt3R package importable (it lives in the ./dust3r folder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dust3r'))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ---- settings ----
CHECKPOINT = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
FRAME_DIR  = 'frames'
FRAME_COUNTS_TO_TRY = [12, 8, 5]   # drops down automatically if the GPU runs out of memory
DEVICE = 'cuda'
# -------------------

def extract_frames(video_path, num_frames):
    os.makedirs(FRAME_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(FRAME_DIR, '*.jpg')):   # clear old frames so runs don't mix
        os.remove(old)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        sys.exit(f"Couldn't read any frames from '{video_path}'. Is the name/path right?")
    indices = [int(total * i / num_frames) for i in range(num_frames)]
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(os.path.join(FRAME_DIR, f'frame_{i:02d}.jpg'), frame)
    cap.release()
    return sorted(glob.glob(os.path.join(FRAME_DIR, '*.jpg')))

def run_dust3r(image_paths, model):
    images = load_images(image_paths, size=512)
    pairs  = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, DEVICE, batch_size=1)
    scene  = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    pts, cols = [], []
    for img, p, m in zip(scene.imgs, scene.get_pts3d(), scene.get_masks()):
        mask = m.cpu().numpy()
        pts.append(p.detach().cpu().numpy()[mask])
        cols.append((img[mask] * 255).astype(np.uint8))
    return np.concatenate(pts), np.concatenate(cols)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python reconstruct.py <video_file>\nExample: python reconstruct.py my_room.mp4")
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        sys.exit(f"Can't find video '{video_path}'. Put it in this folder and check the name.")

    out_ply = os.path.splitext(os.path.basename(video_path))[0] + '.ply'
    print('Loading DUSt3R model...')
    model = AsymmetricCroCo3DStereo.from_pretrained(CHECKPOINT).to(DEVICE)

    for n in FRAME_COUNTS_TO_TRY:
        print(f'\nTrying with {n} frames...')
        frames = extract_frames(video_path, n)
        print(f'Extracted {len(frames)} frames. Running reconstruction (this can take a minute or two)...')
        try:
            points, colors = run_dust3r(frames, model)
            trimesh.PointCloud(vertices=points, colors=colors).export(out_ply)
            print(f'\nDone! Saved {len(points):,} points to {out_ply}')
            print(f'Open {out_ply} in MeshLab to view the 3D scene.')
            return
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f'GPU ran out of memory at {n} frames — trying fewer...')
            else:
                raise   # a different error: let it surface so we can see it

    sys.exit('Ran out of memory even at the lowest frame count. Close other GPU apps and try again.')

if __name__ == '__main__':
    main()