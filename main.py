import torch
import torchaudio
import os
import subprocess
import numpy as np
import shutil

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

class DreamTalkContext:
    def __init__(self, directory):
        self.directory = directory
        self.original_directory = os.getcwd()

    def __enter__(self):
        os.chdir(self.directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original_directory)

# Adapted from https://github.com/ali-vilab/dreamtalk/blob/main/inference_for_demo_video.py
def generate(name, mood, gender, device="cuda"):
    with DreamTalkContext("./dreamtalk"):
        from dreamtalk.inference_for_demo_video import (
            get_cfg_defaults,
            get_diff_net,
            crop_src_image,
            inference_one_video,
            get_netG,
            render_video,
        )

        device = torch.device(device)

        cfg = get_cfg_defaults()
        cfg.CF_GUIDANCE.SCALE = 1.0
        cfg.freeze()

        wav_path = f"../audio/{name}.wav"
        img_path = f"../img/{name}.jpg"
        gender_prefix = "M030" if gender == "male" else "W009"
        supported_moods = [
            "angry",
            "contempt",
            "disgusted",
            "happy",
            "neutral",
            "surprised",
            "sad",
            "fear",
        ]
        if mood not in supported_moods:
            raise NotImplementedError(
                f'Mood "{mood}" was not recognized. Supported moods are {supported_moods}'
            )
        level = "1" if mood == "neutral" else "3"
        style_clip_path = (
            f"./data/style_clip/3DMM/{gender_prefix}_front_{mood}_level{level}_001.mat"
        )

        tmp_dir = f"./tmp/{name}"
        os.makedirs(tmp_dir, exist_ok=True)

        # get audio in 16000Hz
        wav_16k_path = os.path.join(tmp_dir, f"{name}_16K.wav")
        command = f"ffmpeg -y -i {wav_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
        subprocess.run(command.split())

        # get wav2vec feat from audio
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )

        wav2vec_model = (
            Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            .eval()
            .to(device)
        )

        speech_array, sampling_rate = torchaudio.load(wav_16k_path)
        audio_data = speech_array.squeeze().numpy()
        inputs = wav2vec_processor(
            audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            audio_embedding = wav2vec_model(
                inputs.input_values.to(device), return_dict=False
            )[0]

        audio_feat_path = os.path.join(tmp_dir, f"{name}_wav2vec.npy")
        np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

        # get src image
        src_img_path = os.path.join(tmp_dir, "src_img.png")
        crop_src_image(img_path, src_img_path, 0.4)

        with torch.no_grad():
            # get diff model and load checkpoint
            diff_net = get_diff_net(cfg, device).to(device)
            # generate face motion
            face_motion_path = os.path.join(tmp_dir, f"{name}_facemotion.npy")
            inference_one_video(
                cfg,
                audio_feat_path,
                style_clip_path,
                "./data/pose/RichardShelby_front_neutral_level1_001.mat",
                face_motion_path,
                diff_net,
                device,
                max_audio_len=30,
            )
            # get renderer
            renderer = get_netG("./checkpoints/renderer.pt", device)
            # render video
            output_video_path = f"../output/{name}.mp4"
            render_video(
                renderer,
                src_img_path,
                face_motion_path,
                wav_16k_path,
                output_video_path,
                device,
                fps=25,
                no_move=False,
            )
