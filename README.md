# DreamTalk demo
A demo of [DreamTalk](https://github.com/ali-vilab/dreamtalk) from Ma et. al, for the purpose of video generation.

## Installation
Clone this repository with submodules:
```bash
git clone --recurse-submodules <repo_url>
```
### Install python 3.7
At the time of this writing, the default python version in the Hyperstack Ubuntu VM is python 3.10. However, we will need python 3.7 due to specific dependency requirements.

We can install python 3.7 like follows:
```bash
export NEEDRESTART_MODE=a
sudo apt update
sudo apt-get -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo -E apt-get -y install python3.7
sudo -E apt-get -y install python3.7-dev
sudo -E apt-get -y install python3.7-distutils
```

If everything went well, you should be able to run the following command successfully:
```bash
python3.7 --version
```

### Install ffmpeg and related libraries
```bash
sudo -E apt-get -y install libavdevice-dev libavfilter-dev libavformat-dev
sudo -E apt-get -y install ffmpeg
```

### Other system dependencies
```bash
sudo -E apt-get -y install cudnn9-cuda-12
sudo -E apt-get -y install libopenblas-dev liblapack-dev
sudo -E apt-get -y install libx11-dev
sudo -E apt-get -y install pkg-config
sudo -E apt-get -y install cmake
```

### Install project dependencies
We can now install the virtual environment and the dreamtalk dependencies, including PyTorch with GPU acceleration:
```bash
sudo -E apt-get -y install python3-virtualenv
virtualenv --python=python3.7 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To confirm whether you have GPU acceleration enabled or not, run the following command:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
It should output `True`.

Note: there's no need to install the requirements at `./dreamtalk/requirements.txt`, as the relevant dependencies have already been included at the local requirement files.

## Checkpoints

You need to copy the checkpoints manually to the `./dreamtalk/checkpoints` folder. The checkpoints are not publicly available and you need to get them from the authors, as described [here](https://github.com/ali-vilab/dreamtalk?tab=readme-ov-file#download-checkpoints).

## Checking installation

You can verify if you followed all steps successfully by running a sample generation.

When running with GPU acceleration:
```bash
source .venv_gpu/bin/activate
cd dreamtalk
python inference_for_demo_video.py \
--wav_path data/audio/acknowledgement_english.m4a \
--style_clip_path data/style_clip/3DMM/M030_front_neutral_level1_001.mat \
--pose_path data/pose/RichardShelby_front_neutral_level1_001.mat \
--image_path data/src_img/uncropped/male_face.png \
--cfg_scale 1.0 \
--max_gen_len 30 \
--output_name acknowledgement_english@M030_front_neutral_level1_001@male_face
```

When running on the CPU:
```bash
source .venv_cpu/bin/activate
cd dreamtalk
python inference_for_demo_video.py \
--wav_path data/audio/acknowledgement_english.m4a \
--style_clip_path data/style_clip/3DMM/M030_front_neutral_level1_001.mat \
--pose_path data/pose/RichardShelby_front_neutral_level1_001.mat \
--image_path data/src_img/uncropped/male_face.png \
--cfg_scale 1.0 \
--max_gen_len 30 \
--output_name acknowledgement_english@M030_front_neutral_level1_001@male_face
--device=cpu
```

In either case, the output will appear at `./dreamtalk/output_video/acknowledgement_english@M030_front_neutral_level1_001@male_face.mp4` (it will be overritten, if it exists).

## Permitted use

In accordance with the [disclaimer](https://github.com/ali-vilab/dreamtalk?tab=readme-ov-file#disclaimer) in the Dreamtalk repository, the content of this repository is for RESEARCH/NON-COMMERCIAL USE ONLY.