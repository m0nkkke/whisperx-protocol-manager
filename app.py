import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

import whisperx
import torch
import torchaudio
import numpy as np
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import pandas as pd
import google.generativeai as genai
from google.api_core.client_options import ClientOptions

load_dotenv()

# WhisperX параметры
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

WHISPER_MODEL_SIZE          = os.getenv("MODEL_NAME", "large-v3")
WHISPER_COMPUTE_TYPE        = os.getenv("COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE           = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_LENGTH_PENALTY      = float(os.getenv("WHISPER_LENGTH_PENALTY", "1.0"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
DEFAULT_INITIAL_PROMPT      = os.getenv("WHISPER_INITIAL_PROMPT", "")
WHISPER_VAD_CHUNK           = int(os.getenv("WHISPER_VAD_CHUNK", "30"))
WHISPER_VAD_ONSET           = float(os.getenv("WHISPER_VAD_ONSET", "0.4"))
WHISPER_VAD_OFFSET          = float(os.getenv("WHISPER_VAD_OFFSET", "0.6"))
WHISPER_LANGUAGE            = os.getenv("WHISPER_LANGUAGE", "ru")

MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "2"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))
SEG_THRESH   = float(os.getenv("SEGMENTATION_THRESHOLD", "0.60"))
SEG_MIN_OFF  = float(os.getenv("SEGMENTATION_MIN_DURATION_OFF", "0.20"))
CLUST_THRESH = float(os.getenv("CLUSTERING_THRESHOLD", "0.50"))
CLUST_MIN_SZ = int(os.getenv("CLUSTERING_MIN_CLUSTER_SIZE", "8"))

# Настройка Gemini
genai.configure(
    api_key=GEMINI_API_KEY,
    client_options=ClientOptions(api_endpoint="generativelanguage.googleapis.com")
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join("static", "index.html"), "r", encoding="utf-8").read()

def format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def load_reference_embeddings(folder: str, hf_token: str):
    inference = Inference("pyannote/embedding", use_auth_token=hf_token, window="whole")
    voices = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(folder, fname)
        waveform_np = whisperx.load_audio(path)
        waveform = torch.tensor(waveform_np).unsqueeze(0)
        emb_result = inference({"waveform": waveform, "sample_rate": 16000})
        
        # гарантированно получить нужный тензор
        emb_tensor = emb_result["embedding"] if isinstance(emb_result, dict) else emb_result
        voices[os.path.splitext(fname)[0]] = emb_tensor
    return voices


def to_vector(x):
    if isinstance(x, dict):
        if "embedding" in x:
            x = x["embedding"]
        else:
            raise TypeError("Dict embedding missing 'embedding' key")
    if isinstance(x, torch.Tensor):
        return x.squeeze().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x.squeeze()
    else:
        print("❌ Unsupported type:", type(x), "→ content:", x)
        raise TypeError("Unsupported embedding type")



def identify_speaker(embedding, reference_embeddings, threshold=0.75):
    emb_vector = to_vector(embedding)
    best_score = float("inf")
    best_speaker = None
    for name, ref_emb in reference_embeddings.items():
        print(f"[🧪 DEBUG] ref_emb type: {type(ref_emb)}")
        print(f"[🧪 DEBUG] ref_emb content: {ref_emb}")
        ref_vector = to_vector(ref_emb)
        score = cosine(emb_vector, ref_vector)
        if score < best_score:
            best_score = score
            best_speaker = name
    if best_score < threshold:
        return best_speaker
    return None


def transcribe_and_diarize(
    audio_path: str,
    do_diarize: bool,
    user_prompt: str,
    min_speakers: int,
    max_speakers: int,
    device: str,
    compute_type: str,
    model_name: str
) -> str:
    prompt = user_prompt.strip() or DEFAULT_INITIAL_PROMPT

    asr_opts = {
        "beam_size": WHISPER_BEAM_SIZE,
        "length_penalty": WHISPER_LENGTH_PENALTY,
        "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "initial_prompt": prompt,
        "temperatures": [0.0],
    }

    vad_opts = {
        "chunk_size": WHISPER_VAD_CHUNK,
        "vad_onset": WHISPER_VAD_ONSET,
        "vad_offset": WHISPER_VAD_OFFSET,
    }

    print(f"\n📢 WhisperX → Модель: {model_name}, Устройство: {device}, Тип: {compute_type}")

    pipeline = whisperx.load_model(
        whisper_arch=model_name,
        device=device,
        compute_type=compute_type,
        asr_options=asr_opts,
        vad_method="pyannote",
        vad_options=vad_opts,
        language=WHISPER_LANGUAGE
    )

    if pipeline is None:
        raise RuntimeError("Не удалось загрузить WhisperX!")

    result = pipeline.transcribe(audio_path, batch_size=6, print_progress=False)
    segments = result["segments"]

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
    aligned = whisperx.align(segments, model_a, metadata, whisperx.load_audio(audio_path), "cpu")
    segments = aligned["segments"]

    if do_diarize:
        pyannote = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=HF_TOKEN
        ).instantiate({
            "segmentation": {
                "threshold": SEG_THRESH,
                "min_duration_off": SEG_MIN_OFF
            },
            "clustering": {
                "threshold": CLUST_THRESH,
                "min_cluster_size": CLUST_MIN_SZ
            }
        })
        diar = pyannote(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

        diar_df = pd.DataFrame([
            {"start": t.start, "end": t.end, "speaker": spk}
            for t, _, spk in diar.itertracks(yield_label=True)
        ])
        reference_embeddings = load_reference_embeddings("reference_voices", HF_TOKEN)
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        inference = Inference(model, window="whole")

        # загружаем всё аудио один раз
        waveform, sample_rate = torchaudio.load(audio_path)

        speaker_map = {}
        spk_counter = 0
        MIN_SAMPLES = 16000
        for t, _, spk in diar.itertracks(yield_label=True):
            start_sample = int(t.start * sample_rate)
            end_sample = int(t.end * sample_rate)
            if end_sample - start_sample < MIN_SAMPLES:
                print(f"[⚠] Пропущен короткий сегмент {t}")
                continue
            segment_audio = waveform[:, start_sample:end_sample].cpu()

            emb_result = inference({'waveform': segment_audio, 'sample_rate': sample_rate})
            emb_tensor = emb_result["embedding"] if isinstance(emb_result, dict) else emb_result

            known_speaker = identify_speaker(emb_tensor, reference_embeddings)

            if known_speaker:
                speaker_map[spk] = known_speaker
            else:
                if spk not in speaker_map:
                    speaker_map[spk] = f"SPEAKER_{spk_counter:02d}"
                    spk_counter += 1

        diar_df["speaker"] = diar_df["speaker"].map(speaker_map)
        segments = whisperx.assign_word_speakers(diar_df, {"segments": segments})["segments"]

    lines = []
    for seg in segments:
        ts = f"[{format_ts(seg['start'])}-{format_ts(seg['end'])}]"
        text = seg["text"].strip()
        if do_diarize and "speaker" in seg:
            lines.append(f"{seg['speaker']} {ts} – {text}")
        else:
            lines.append(f"{ts} – {text}")
    return "\n".join(lines)


def generate_meeting_minutes(transcribed_text: str, style: str = "default") -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt_path = os.path.join("prompts", f"{style}.txt")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join("prompts", "default.txt")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    full_prompt = template.replace("{{TEXT}}", transcribed_text)
    response = model.generate_content(full_prompt)
    return response.text.strip()


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    initial_prompt: str = Form(""),
    protocol_prompt_style: str = Form("default"),
    min_speakers: int = Form(2),
    max_speakers: int = Form(2),
    device: str = Form("cpu"),
    compute_type: str = Form("int8"),
    model_name: str = Form("large-v3")
):
    orig_ext = os.path.splitext(file.filename)[-1].lower()
    raw_path = "tmp_input" + orig_ext
    wav_path = "tmp_input.wav"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        transcript = transcribe_and_diarize(
            wav_path, diarize, initial_prompt,
            min_speakers, max_speakers,
            device=device,
            compute_type=compute_type,
            model_name=model_name
        )

        protocol = generate_meeting_minutes(transcript, style=protocol_prompt_style)

    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return JSONResponse({
        "transcript": transcript,
        "protocol": protocol
    })
