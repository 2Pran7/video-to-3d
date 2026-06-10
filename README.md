# video-to-3d

Reconstruct a geometrically coherent 3D point cloud of an indoor room from a short phone video, using the **DUSt3R** feed-forward model. Built as a take-home for the Humanoid *Perception & Spatial AI* internship.

[3D reconstruction of a bedroom](examples/result.png)

*A bedroom reconstructed from a ~55-second handheld phone video — bed, headboard, radiator and walls all recovered at sensible depths.*

**[Sample input video (Google Drive)](https://drive.google.com/file/d/10ZTSDNGz4ZBadH5MFgNXpLJ6TDTsNp0X/view?usp=sharing)** — the original phone footage this reconstruction was built from.
=======

See the [`examples/`](examples/) folder for sample input frames and multiple viewing angles of the output.

## What it does

Point it at a short phone video of a room and it returns a coloured 3D point cloud (`.ply`) you can open in MeshLab or any 3D viewer. One command, any video:

```bash
python reconstruct.py my_room.mp4
```

## How it works

The pipeline turns a video into 3D in four conceptual stages:

1. **Video → frames.** The video (e.g. ~55 seconds, several hundred frames) is sampled down to a handful of evenly-spaced still images. We deliberately keep this small (≈12 frames) — DUSt3R is memory-hungry, and on a 6GB GPU feeding it everything would run out of memory. Evenly-spaced sampling also ensures the frames span the full camera path rather than clustering in one moment.
2. **Pairing.** DUSt3R reasons about *two images at a time*, so the frames are formed into overlapping pairs.
3. **Feed-forward inference.** For each pair, the network directly predicts a "pointmap" — a 3D point for every pixel — plus a per-pixel confidence. This is what makes DUSt3R different from classical Structure-from-Motion: there's no separate feature-matching, camera-calibration or triangulation stage; the network learns to do it end-to-end.
4. **Global alignment.** Each pair is predicted in its own coordinate frame. A final optimisation step solves for the camera poses and a single shared coordinate system so that all the overlapping pairs agree — like fitting puzzle pieces until the edges line up. Low-confidence points are filtered out, and the rest are exported as a coloured point cloud.

## Why DUSt3R

I treated this as a **systems-integration problem, not a research one**. I surveyed the landscape — classical SfM (e.g. COLMAP), NeRF / Gaussian Splatting, and feed-forward models like DUSt3R and VGGT — and chose DUSt3R for three reasons:

- **Hardware fit.** I developed this on a 6GB laptop GPU (RTX 3060). DUSt3R is the lightest option that still produces coherent geometry; NeRF/Gaussian Splatting are heavier and more focused on novel-view rendering than on the geometry itself.
- **Conceptual clarity.** As someone new to 3D vision, a single feed-forward model I could actually understand and explain was worth more than a multi-stage classical pipeline I'd be operating as a black box.
- **Robustness to casual capture.** It needs no camera intrinsics or careful calibration, which suits a "short phone video" input.

## Setup

**Prerequisites:** Python 3.11, an NVIDIA GPU with CUDA, and Git.

1. **Clone this repo and DUSt3R into it:**
```bash
   git clone https://github.com/2Pran7/video-to-3d.git
   cd video-to-3d
   git clone https://github.com/naver/dust3r.git
   cd dust3r && git submodule update --init --recursive && cd ..
```
2. **Download the model checkpoint** `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` (linked from the DUSt3R repo) into a `checkpoints/` folder.
3. **Create a virtual environment and install dependencies:**
```bash
   py -3.11 -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
```
4. **Install the CUDA build of PyTorch** (see *Challenges* below for why this matters):
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

## Usage

```bash
python reconstruct.py your_video.mp4
```

The tool extracts frames, runs DUSt3R, and saves `your_video.ply`. If the GPU runs out of memory it automatically retries with fewer frames (12 → 8 → 5), so it works across different hardware without editing any code. Open the resulting `.ply` in [MeshLab](https://www.meshlab.net/) to view.

## Design decisions & challenges

- **One command, any video.** I merged frame-extraction and reconstruction into a single tool that takes the video as an argument and auto-handles GPU memory limits, so a reviewer can run it on their own footage without touching the code. Usability was an explicit goal.
- **Dependency conflicts.** DUSt3R's install silently replaced my CUDA PyTorch with a CPU-only build (which would run inference on the CPU, painfully slowly). I diagnosed this and force-reinstalled the CUDA build, and pinned the handful of package versions that otherwise clashed — captured in `requirements.txt` so the environment is reproducible.
- **Cut the Gradio demo.** DUSt3R ships an interactive web UI that kept crashing on my machine. Since it wasn't the deliverable, I dropped it and wrote a direct script instead — focusing effort on the actual output rather than a distraction.
- **Camera motion matters more than frame count.** While testing, I compared a clip where I stood still and rotated against one where I physically moved around the scene. The stood-still version warped badly; the moving version was clean. This is *parallax* — the same reason two eyes give depth perception. Changing viewpoint, not adding frames, is what gives the model real depth information.

## Limitations & future work

- **Geometry depends on parallax.** Best results need the camera to physically move through the scene. Pure rotation produces flat, warped output.
- **Output is a point cloud, not a mesh.** Surface reconstruction (e.g. Poisson meshing) would be a natural next step.
- **Scale and large scenes.** The 6GB VRAM ceiling caps frame count, limiting very large rooms.
- **Semantic labels (bonus).** I scoped the optional semantic-labelling bonus (e.g. tagging chairs/tables) as future work — the approach would be to run a 2D segmentation model on the frames and lift the labels onto the 3D points using DUSt3R's pixel-to-point correspondence.

## Credits

This project uses [**DUSt3R**](https://github.com/naver/dust3r) by NAVER Labs Europe, used under its original license. The contribution here is the end-to-end pipeline, the one-command tool, and the design decisions around running it on modest hardware.
