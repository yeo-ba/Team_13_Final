# BLDSAG: Mask-Guided Image Editing with Diffusion Models

This repository implements a custom image-editing pipeline that combines **Stable Diffusion 2.1**, **ControlNet Canny conditioning**, **SAM-HQ mask generation**, **Blended Latent Diffusion**, **Classifier-Free Guidance**, and **Self-Attention Guidance**. The project focuses on preserving the structure of an input image while editing a user-selected region with a text prompt.

## Why This Project Matters

Modern text-to-image models can produce high-quality images, but localized editing is harder: the edited object should follow the prompt while the untouched background should remain stable. This project tackles that problem by combining segmentation, edge conditioning, attention-guided denoising, and latent-space blending in one inference workflow.

## Key Features

- **Region-aware editing**: uses SAM-HQ to generate a mask from a bounding-box prompt.
- **Structure preservation**: extracts Canny edges from the source image and feeds them to ControlNet.
- **Localized latent blending**: keeps the original background latent outside the mask after a configurable denoising step.
- **Self-Attention Guidance**: stores UNet attention probabilities and degrades high-attention regions to improve generation focus.
- **Reproducible generation**: supports deterministic output through manual seeds.
- **Notebook workflow**: includes an end-to-end Jupyter notebook for mask creation, pipeline loading, generation, and visualization.

## Repository Structure

```text
.
├── BLDSAG1024.py   # Core pipeline, SAM-HQ preprocessing, grid visualization utilities
├── BLDSAG.ipynb    # Example notebook for end-to-end image editing
├── img.png         # Example source image
├── requirements.txt
├── .gitignore
└── README.md
```

## Technical Overview

The pipeline runs in four main stages:

1. **Mask selection**
   - `ImageGrid` overlays a 16 x 16 coordinate grid on the input image.
   - `SamHQImageProcessor` converts `(x1, y1, x2, y2)` grid coordinates into a SAM-HQ bounding box.
   - The selected mask is saved as `mask.png`, and the masked source image is saved as `image.png`.

2. **Condition preparation**
   - The source image is resized to 1024 x 1024.
   - Canny edges are generated and saved as `canny.png`.
   - The edge map is used as ControlNet conditioning.

3. **Diffusion inference**
   - Stable Diffusion 2.1 predicts noise with classifier-free guidance.
   - ControlNet injects structural guidance from the Canny map.
   - Self-Attention Guidance uses stored mid-block attention maps to create degraded latents and refine the denoising direction.

4. **Latent blending**
   - After `blending_start_percentage`, the pipeline blends generated latents inside the mask with noised source latents outside the mask.
   - This preserves the original background while allowing prompt-driven edits in the selected region.

## Requirements

This project is designed for a CUDA-enabled environment. The notebook was originally run on a multi-GPU research server, so update device IDs before running locally.

Core dependencies:

```bash
pip install torch torchvision diffusers transformers accelerate opencv-python pillow matplotlib numpy
```

SAM-HQ is also required:

```bash
git clone https://github.com/SysCV/sam-hq.git
pip install -e sam-hq
mkdir -p pretrained_checkpoint
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth -O pretrained_checkpoint/sam_hq_vit_h.pth
```

Model checkpoints used by the example:

- `stabilityai/stable-diffusion-2-1`
- `thibaud/controlnet-sd21-canny-diffusers`
- `pretrained_checkpoint/sam_hq_vit_h.pth`

## Quick Start

1. Open `BLDSAG.ipynb`.
2. Update the image path:

```python
image_path = "img.png"
```

3. Update device IDs for your machine:

```python
SAM_device = "cuda:0"
BLDCSAG_device = "cuda:0"
```

4. Use the grid helper to choose a mask region:

```python
image_grid = ImageGrid(image_path)
image_grid.draw_grid()
```

5. Generate the mask and masked input:

```python
img_processor = SamHQImageProcessor(
    img_path=image_path,
    device=SAM_device,
    coord=(4, 6, 12, 15),
)
img_processor.run()
```

6. Load ControlNet and run the editing pipeline:

```python
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-sd21-canny-diffusers",
    torch_dtype=torch.float16,
    cross_attention_dim=1024,
)

pipe = BLDCSAG1024(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    prompt="They are sitting in Central Park in New York City",
    negative_prompt="low quality, blurry, distorted, out of focus, low resolution",
    blending_start_percentage=0.1,
    device=BLDCSAG_device,
)

results = pipe.generate_image(
    kernel_size=2,
    num_inference_steps=50,
    guidance_scale=7.0,
    sag_scale=0.7,
    seed=1998,
)
```

## Main Classes

- `BLDCSAG1024`: custom Stable Diffusion + ControlNet pipeline with SAG and latent blending.
- `CrossAttnStoreProcessor`: attention processor that stores UNet attention probabilities for SAG.
- `SamHQImageProcessor`: creates `mask.png` and `image.png` from SAM-HQ segmentation.
- `ImageGrid`: displays a 16 x 16 coordinate grid for mask prompt selection.
- `ImageGridDisplay`: visualizes the original image, mask, Canny map, and final output.

## Engineering Highlights

- Implemented a custom inference loop instead of relying only on high-level Diffusers pipeline calls.
- Integrated multiple model components with different conditioning signals: text, Canny edges, segmentation masks, and attention maps.
- Managed latent-space transformations manually, including VAE encode/decode, scheduler stepping, noise addition, and mask resizing.
- Added deterministic generation controls for easier debugging and result comparison.
- Built notebook utilities to make the experimental workflow easier to inspect and iterate on.

## Known Limitations

- The example notebook contains machine-specific GPU settings and should be adjusted before running on a new environment.
- The pipeline assumes 1024 x 1024 generation and uses `height // 8`, `width // 8` latent dimensions.
- Full execution requires large model downloads and a CUDA GPU with enough memory for Stable Diffusion 2.1, ControlNet, and SAM-HQ.

## Future Improvements

- Add a CLI wrapper for repeatable batch experiments.
- Move model paths, device IDs, and generation parameters into a config file.
- Add automated checks for missing checkpoints and invalid mask coordinates.
