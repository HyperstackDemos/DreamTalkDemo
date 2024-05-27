import os

import torch
from melo.api import TTS


class OpenVoiceContext:
    def __init__(self):
        self.directory = "./OpenVoice"
        self.original_directory = os.getcwd()

    def __enter__(self):
        os.chdir(self.directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original_directory)


language_to_speaker_key_mapping = {
    "EN": "EN-US",
    "ES": "ES",
    "FR": "FR",
    "ZH": "ZH",
    "JP": "JP",
    "KR": "KR",
}


# Adapted from https://github.com/myshell-ai/OpenVoice/blob/main/demo_part3.ipynb
def generate(name, text, language_code, device="cuda"):
    with OpenVoiceContext():
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        if language_code not in ["EN", "ES", "FR", "ZH", "JP", "KR"]:
            raise NotImplementedError(f"Unsupported language code:{language_code}")

        ckpt_converter = "./checkpoints_v2/converter"
        output_dir = "../output"

        tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", device=device
        )
        tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        os.makedirs(output_dir, exist_ok=True)

        reference_speaker = f"../audio/{name}.wav"
        target_se, audio_name = se_extractor.get_se(
            reference_speaker, tone_color_converter, vad=False
        )

        src_path = f"{output_dir}/tmp.wav"

        speed = 1.0

        model = TTS(language=language_code, device=device)
        speaker_key = language_to_speaker_key_mapping[language_code]
        speaker_id = model.hps.data.spk2id[speaker_key]
        speaker_file = speaker_key.lower().replace("_", "-")

        source_se = torch.load(
            f"checkpoints_v2/base_speakers/ses/{speaker_file}.pth", map_location=device
        )
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f"../audio/{name}_from_text.wav"

        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
        )
