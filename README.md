# DreamTalk demo

A demo of [DreamTalk](https://github.com/ali-vilab/dreamtalk) from Ma et. al, for the purpose of video generation.

## Installation

Clone this repository with submodules:
```bash
git clone --recurse-submodules <repo_url>
```

At the time of this writing, the default python version in the Hyperstack Ubuntu VM is python 3.10. However, we will need python 3.7 due to specific dependency requirements.

We can install python 3.7 like follows:

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
sudo apt install python3.7-distutils
```

If everything went well, you should be able to run the following command successfully:
```bash
python3.7 --version
```

We can now install the virtual environment and the dependencies:
```bash
sudo apt install python3-virtualenv
virtualenv --python=python3.7 .venv
source .venv/bin/activate
pip install -r ./dreamtalk/requirements.txt
```

## Checkpoints

You need to copy the checkpoints manually to the `./dreamtalk/checkpoints` folder. The checkpoints are not publicly available and you need to get them from the authors, as described [here](https://github.com/ali-vilab/dreamtalk?tab=readme-ov-file#download-checkpoints).