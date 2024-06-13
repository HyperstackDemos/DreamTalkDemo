import argparse
import logging
import os
import sys
import time
from datetime import timedelta

from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


class VideoGenerator:
    def __init__(self, base_dir="batch", device="cuda"):
        self.base_dir = base_dir
        self.device = device

    def setup_asset_generators(self):
        import multiprocessing

        import torch
        from datasets import load_dataset
        from diffusers import StableDiffusion3Pipeline
        from dotenv import load_dotenv
        from huggingface_hub import login

        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        CACHE_DIR = os.getenv("CACHE_DIR")
        login(token=HF_TOKEN)

        # Load voices dataset
        common_voice_en = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "en",
            split="test",
            cache_dir=CACHE_DIR,
            num_proc=multiprocessing.cpu_count(),
        )
        self.male_voices = common_voice_en.filter(
            lambda x: x["gender"] == "male_masculine",
            num_proc=multiprocessing.cpu_count(),
        )
        self.female_voices = common_voice_en.filter(
            lambda x: x["gender"] == "female_feminine",
            num_proc=multiprocessing.cpu_count(),
        )

        # Load text-to-image model
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to(self.device)

    def _generate_audio(self, idx, gender):
        from scipy.io import wavfile

        dataset = self.male_voices if gender == "male" else self.female_voices
        row = dataset[idx]
        audio = row["audio"]["array"]
        sampling_rate = row["audio"]["sampling_rate"]
        file_path = os.path.join(self.base_dir, f"audio/{str(idx).zfill(3)}.wav")
        wavfile.write(file_path, sampling_rate, audio)

    def _generate_person(self, idx, gender):
        prompt = (
            "adult " + "man"
            if gender == "male"
            else "woman"
            + " looking at the camera, casually dressed, flat background, uniform lighting"
        )
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        ).images[0]
        image.save(os.path.join(self.base_dir, f"img/{str(idx).zfill(3)}.jpg"))

    def generate_assets(self, num_generations):
        logging.info(
            f"Starting generation of {num_generations} assets (audio and video)..."
        )
        genders = ["male", "female"]
        for idx in tqdm(range(num_generations)):
            gender = genders[idx % len(genders)]
            self._generate_audio(idx, gender)
            self._generate_person(idx, gender)

    def generate_videos(self, num_generations):
        from video_generator import generate

        audio_dir = os.path.join(self.base_dir, "audio")
        num_failures = 0

        logging.info(f"Starting generation of {num_generations} videos from assets..")
        idx = 0
        for audio_file in tqdm(os.listdir(audio_dir)):
            if idx == num_generations:
                break
            gender = "male" if "male" in audio_file else "female"
            try:
                generate(
                    image_name=audio_file.split(".")[0],
                    audio_name=audio_file.split(".")[0],
                    mood=None,
                    gender=gender,
                    base_dir=os.path.join("..", self.base_dir),
                    device=self.device,
                )
            except Exception as e:
                num_failures += 1
                logging.warning(f"Failed generation at index {idx} with error: {e}")
            idx += 1

        logging.info(f"Proccessed {idx} videos ({num_failures} failures)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="assets")
    parser.add_argument("--num-generations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.mode not in ["assets", "video"]:
        raise NotImplementedError('Mode needs to be "assets" or "video"')

    generator = VideoGenerator(device=args.device)

    start_time = time.time()

    if args.mode == "assets":
        generator.setup_asset_generators()
        generator.generate_assets(args.num_generations)
    else:
        generator.generate_videos(args.num_generations)

    stop_time = time.time()
    total_time = timedelta(seconds=stop_time - start_time)
    logging.info(f"Total time: {str(total_time).split('.')[0]}")
