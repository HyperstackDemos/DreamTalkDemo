# Video generation demo

This repo is a showcase of image-to-video, text-to-speech, and text-to-image tasks for the purpose of video content generation.

In particular, we have a look at the following technologies:
* [DreamTalk](https://github.com/ali-vilab/dreamtalk) from Ma et. al.
* [OpenVoice](https://github.com/myshell-ai/OpenVoice) from Qin et. al.
* [StableDiffusion 3](https://stability.ai/news/stable-diffusion-3) from Stability.AI

Please, refer to the notebooks for an ellaboration on the experiments.


## Installation
You can reproduce the results on a [Hyperstack](https://www.hyperstack.cloud/) VM with ubuntu.

Start by cloning this repository with submodules:
```bash
git clone --recurse-submodules <repo_url>
```

## Install python 3.7
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

## Install ffmpeg and related libraries
```bash
sudo -E apt-get -y install libavdevice-dev libavfilter-dev libavformat-dev
sudo -E apt-get -y install ffmpeg
```

## Other system dependencies
```bash
sudo -E apt-get -y install cudnn9-cuda-12
sudo -E apt-get -y install libopenblas-dev liblapack-dev
sudo -E apt-get -y install libx11-dev
sudo -E apt-get -y install pkg-config
sudo -E apt-get -y install cmake
```

## Virtual enviroments
We will create 2 different virtual environments to contain the different dependencies:
* `venv_dreamtalk`: contains the dependencies needed to run DreamTalk, based on python 3.7.
* `venv_openvoice`: contains the dependencies needed to run OpenVoice and StableDiffusion 3, based on python 3.10.

## Install project dependencies
Let's start with `venv_dreamtalk` virtual environment.

```bash
sudo -E apt-get -y install python3-virtualenv
virtualenv --python=python3.7 .venv_dreamtalk
source .venv_dreamtalk/bin/activate
pip install -r requirements-dreamtalk.txt
```

To confirm whether you have GPU acceleration enabled or not, run the following command:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
It should output `True`.

Let's continue now with the `venv_openvoice` virtual environment. We need to install the set of dependencies for OpenVoice and Mello, which provides text-to-speech capabilities.

```bash
deactivate
virtualenv .venv_openvoice
source .venv_openvoice/bin/activate
cd MeloTTS
pip install -e .
python -m unidic download
echo `pwd` > /home/ubuntu/demos/dreamtalk-demo/.venv_openvoice/lib/python3.10/site-packages/pythonpath.pth
cd ..
pip install -r requirements-openvoice.txt
```

Finally, we need to make the CUDNN libraries available to the Mello package as follows:
```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib
echo "export CUDNN_PATH=$CUDNN_PATH" >> .venv_openvoice/bin/activate
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .venv_openvoice/bin/activate
```

Deactivate and activate again the environment, if needed.

## Checkpoints

You need to copy the checkpoints for DreamTalk to the `./dreamtalk/checkpoints` folder manually. The checkpoints are not publicly available and you need to get them from the authors, as described [here](https://github.com/ali-vilab/dreamtalk?tab=readme-ov-file#download-checkpoints).

## Permitted use

In accordance with the [disclaimer](https://github.com/ali-vilab/dreamtalk?tab=readme-ov-file#disclaimer) in the Dreamtalk repository, the content of this repository is for RESEARCH/NON-COMMERCIAL USE ONLY.