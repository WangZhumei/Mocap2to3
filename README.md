# Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining
### [Project Page](https://wangzhumei.github.io/mocap-2-to-3) | [Paper](https://arxiv.org/abs/2503.03222)
> [Zhumei Wang](https://wangzhumei.github.io/zhumeiwang/)<sup>\*</sup>,
> [Zechen Hu](https://wangzhumei.github.io/mocap-2-to-3)<sup>\*</sup>,
> [Ruoxi Guo](https://www.researchgate.net/profile/Ruoxi-Guo-2),
> [Huaijin Pi](https://phj128.github.io/),
> [Ziyong Feng](https://wangzhumei.github.io/mocap-2-to-3),
> [Liang Zhang](https://wangzhumei.github.io/mocap-2-to-3),
> [Mingtao Pei](https://wangzhumei.github.io/mocap-2-to-3),
> [Siyuan Huang](https://siyuanhuang.com/)<br>
> **CVPR 2026**

<p align="center">
    <img src=docs/image/teaser.png />
</p>


## Setup

### Installation

```bash
conda create -y -n mocap2to3 python=3.10
conda activate mocap2to3
```

Install PyTorch separately so the wheel matches your local CUDA environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the Python dependencies and the project package:

```bash
pip install -r requirements.txt
pip install -e .
```

PyTorch3D should be compiled from source against the active PyTorch/CUDA environment.

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout 2d4d345b6fd2720580bff5f63dcbd3b230b43996

pip install -U ninja cmake fvcore iopath
pip install -v -e . --no-build-isolation
cd ..
```

If you want to render `.npy` motion results into videos, install Blender separately.


```bash
mkdir -p ~/software
cd ~/software

wget https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz
tar -xf blender-4.4.3-linux-x64.tar.xz

~/software/blender-4.4.3-linux-x64/blender --version
```

To make the `blender` command available in new shells, add it to `PATH`:

```bash
echo 'export PATH="$HOME/software/blender-4.4.3-linux-x64:$PATH"' >> ~/.bashrc
source ~/.bashrc
```



## Evaluation
### Required files
Project assets are hosted at:
https://huggingface.co/wzm0217/Mocap2-to-3

Before evaluation, download the following files and place them under the same directory structure used in this repository.

Common assets:

- `inputs/RICH.tar.gz`
  Extract to `inputs/RICH/`
- `resource/rich.tar.gz`
  Extract to `hmr4d/dataset/rich/resource/`
- `inputs/checkpoints/huggingface/clip-vit-base-patch32.tar.gz`
  Extract to `inputs/checkpoints/huggingface/`

Evaluation checkpoints:

- `checkpoints/last.ckpt`
  Save to `outputs/2dmotion_offset_richcam/mdm-smpl_rich/checkpoints/last.ckpt`
- `checkpoints/best.ckpt`
  Save to `outputs/2dmotionmv_persp_richcam_offset/mdm-smpl_mv/checkpoints/best.ckpt`

The expected layout is:

```text
inputs/
├── RICH/
└── checkpoints/
    └── huggingface/
        └── clip-vit-base-patch32/
hmr4d/
└── dataset/
    └── rich/
        └── resource/
outputs/
├── 2dmotion_offset_richcam/
│   └── mdm-smpl_rich/
│       └── checkpoints/
│           └── last.ckpt
└── 2dmotionmv_persp_richcam_offset/
    └── mdm-smpl_mv/
        └── checkpoints/
            └── best.ckpt
```

### Evaluate the final model on RICH
```bash
HYDRA_FULL_ERROR=1 python tools/train.py exp=mas_offset/rich_motion2dmv/mdm global/task=motion2dmv_offset/test2dmv "ckpt_path=[outputs/2dmotion_offset_richcam/mdm-smpl_rich/checkpoints/last.ckpt,outputs/2dmotionmv_persp_richcam_offset/mdm-smpl_mv/checkpoints/best.ckpt]" ckpt_type=pl_2d_mv model.pipeline.args.guidance_scale=1
```

The test callback saves the prediction results under `./res/rich_smpl.pth`.



## Training
### Required files
Project assets are hosted at:
https://huggingface.co/wzm0217/Mocap2-to-3

Training uses the same common assets as evaluation, plus the following training datasets:

- `inputs/hml3d.tar.gz`
  Extract to `inputs/hml3d/`
- `inputs/bedlam.tar.gz`
  Extract to `inputs/bedlam/`
- `inputs/amass.tar.gz`
  Extract to `inputs/amass/`
- `inputs/h36m.tar.gz`
  Extract to `inputs/h36m/`

Multi-view finetuning also requires the neutral SMPL-X model file:

- Register and download the official SMPL-X model from:
  https://smpl-x.is.tue.mpg.de
- After downloading the model package, copy `SMPLX_NEUTRAL.npz` to:
  `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`

This matches the directory layout expected by the `smplx` loader and by this repository.

The expected training layout is:

```text
inputs/
├── RICH/
├── amass/
├── bedlam/
├── h36m/
├── hml3d/
└── checkpoints/
    ├── body_models/
    │   └── smplx/
    │       └── SMPLX_NEUTRAL.npz
    └── huggingface/
        └── clip-vit-base-patch32/
hmr4d/
└── dataset/
    └── rich/
        └── resource/
```


### Stage 1: pretrain on 2D data

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HYDRA_FULL_ERROR=1 python tools/train.py exp=mas_offset/motion2d/mdm pl_trainer.devices=8
```

### Stage 2: multi-view finetuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HYDRA_FULL_ERROR=1 python tools/train.py exp=mas_offset/motion2dmv/mdm ckpt_path=outputs/2dmotion_offset_richcam/mdm-smpl_rich/checkpoints/last.ckpt ckpt_type=pl data=motion2dmv_offset/HumanML3D_2dmv pl_trainer.devices=8
```


## Visualization

To visualize the prediction results saved in `./res`, first fit SMPLH vertices for each sub-sequence, then render the saved mesh `.npy` files into videos.

### Step 1: fit SMPLH vertices

```bash
python tools/mas/joints2smpl.py \
  --input ./res/your_saved_prediction.pth \
  --output-dir ./render_result/npy/mocap-2-to-3 \
  --device 0
```

This command reads the predicted joint motion from a saved `.pth` file under `./res/` and saves fitted mesh sequences under `./render_result/npy/mocap-2-to-3/`.

### Step 2: render videos

```bash
python tools/mas/render_motion.py \
  --dir ./render_result/npy/mocap-2-to-3 \
  --output-dir ./render_result/video/mocap-2-to-3 \
  --mode video \
  --joint-type HumanML3D \
  --device 0
```

This command renders every mesh `.npy` file in the folder and saves the final videos under `./render_result/video/mocap-2-to-3/`.




# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@InProceedings{Wang_2026_CVPR,
    author    = {Wang, Zhumei and Hu, Zechen and Guo, Ruoxi and Pi, Huaijin and Feng, Ziyong and Zhang, Liang and Pei, Mingtao and Huang, Siyuan},
    title     = {Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={42869--42878},
    year={2026}
}
```
