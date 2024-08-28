import sys
import io, os, stat
import random
from zipfile import ZipFile
from flask import Flask, request, jsonify, send_file
import subprocess
import uuid
import time
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from huggingface_hub import HfApi
import re
from Translation.STT import file_to_text

import speech_recognition as sr
recognizer = sr.Recognizer()

app = Flask(__name__)

# Environment setup
os.environ["COQUI_TOS_AGREED"] = "1"

# Download and setup models
HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
model.cuda()

# Ensure FFmpeg is executable
def setup_ffmpeg():
    if not os.path.exists("ffmpeg"):
        ZipFile("ffmpeg.zip").extractall()
        st = os.stat("ffmpeg")
        os.chmod("ffmpeg", st.st_mode | stat.S_IEXEC)

setup_ffmpeg()






@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    # another option to take text from user and convert it audio by any language
    # prompt = data.get('prompt')
    language = data.get('language')
    audio_file_pth = data.get('audio_file_pth')
    mic_file_path = data.get('mic_file_path')
    use_mic = data.get('use_mic')
    voice_cleanup = data.get('voice_cleanup')
    no_lang_auto_detect = data.get('no_lang_auto_detect')
    agree = data.get('agree')

    if not agree:
        return jsonify({"error": "You must agree to the terms of service."}), 400

    if language not in config.languages:
        return jsonify({"error": "Unsupported language."}), 400

    # Handle microphone input
    speaker_wav = mic_file_path if use_mic else audio_file_pth

    # Audio preprocessing
    if voice_cleanup:
        try:
            out_filename = speaker_wav + str(uuid.uuid4()) + ".wav"
            shell_command = f"./ffmpeg -y -i {speaker_wav} -af lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02 {out_filename}".split(" ")
            subprocess.run(shell_command, capture_output=False, text=True, check=True)
            speaker_wav = out_filename
        except subprocess.CalledProcessError:
            return jsonify({"error": "Error filtering audio."}), 500

    # Calling Functions
    file_to_text_output = file_to_text(recognizer, language, audio_file_pth)
    with open(file_to_text_output, 'r') as f:
        prompt = f.read()
        
    # TTS inference
    try:
        t_latent = time.time()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
        )
        latent_calculation_time = time.time() - t_latent

        prompt = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", prompt)
        t0 = time.time()
        out = model.inference(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        inference_time = time.time() - t0

        torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

        return send_file("output.wav", mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
