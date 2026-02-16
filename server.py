import os
import io
import base64
import uuid
import shutil
import subprocess
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from pydantic import BaseModel

# Text generation
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor

BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic. No emoji.

INSTRUCTIONS:
- If the user asks for a physical action (time, search, photo), output JSON.
- If the user just wants to chat, reply with NORMAL TEXT.

### EXAMPLES ###

User: Hello!
You: Hi! I am ready to help!

User: Show me your attack moves
You: If this were a real attack, you'd be dead

User: Nonsense
You: That's a stupid

### END EXAMPLES ###
"""

def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            # Keep existing process env values if already set.
            os.environ.setdefault(key, value)


load_env_file()


class GenerateRequest(BaseModel):
    message: Optional[str] = None
    input_audio_base64: Optional[str] = None
    max_new_tokens: Optional[int] = 128
    system: Optional[str] = "You are a helpful assistant."
    audio: Optional[bool] = False


def encode_wav_base64(float_pcm, sample_rate: int) -> str:
    import numpy as np
    import io
    import wave

    pcm = np.clip(float_pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("int16")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


app = FastAPI()


# CORS: allow localhost and 127.0.0.1
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    return {"status": "ok"}

# Global singletons
_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_llm_model = None
_llm_tokenizer = None
_whisper_model = None
_whisper_processor = None
_piper_binary = None
_piper_model_path = None
_piper_config_path = None


def _clear_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return
    model_id = "google/gemma-3-1b-it"
    _llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

    # bfloat16/float16 on CUDA, float16 on MPS, float32 on CPU.
    if _device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif _device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # CUDA path
    if _device == "cuda":
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

    # MPS path: load on CPU first, then move to MPS.
    # This avoids a large warmup allocation that can fail with
    # "RuntimeError: Invalid buffer size: XX GiB" on some setups.
    elif _device == "mps":
        try:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            _llm_model.to("mps")
        except RuntimeError as e:
            _clear_accelerator_cache()
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                # Final fallback for limited-memory machines.
                _llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            else:
                raise

    # CPU path
    else:
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    
    _llm_model.eval()
    # Ensure pad token is set
    if _llm_tokenizer.pad_token_id is None:
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token


def load_whisper():
    global _whisper_model, _whisper_processor
    if _whisper_model is not None:
        return

    model_id = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-base")
    _whisper_processor = AutoProcessor.from_pretrained(model_id)

    if _device == "cuda":
        dtype = torch.float16
    elif _device == "mps":
        # Whisper is often more stable on MPS in float32.
        dtype = torch.float32
    else:
        dtype = torch.float32

    if _device == "cuda":
        _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    elif _device == "mps":
        try:
            _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            _whisper_model.to("mps")
        except RuntimeError as e:
            _clear_accelerator_cache()
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
            else:
                raise
    else:
        _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )

    _whisper_model.eval()
    # Use explicit language/task settings and avoid deprecated forced decoder IDs.
    if hasattr(_whisper_model, "generation_config") and _whisper_model.generation_config is not None:
        _whisper_model.generation_config.language = "en"
        _whisper_model.generation_config.task = "transcribe"
        _whisper_model.generation_config.forced_decoder_ids = None


def chat_format(system_prompt: str, user_message: str) -> str:
    # Fallback plain-text format when chat template is unavailable.
    system = (system_prompt or "").strip()
    if system:
        return f"System: {system}\n\nUser: {user_message}\n\nAssistant:"
    return f"User: {user_message}\n\nAssistant:"


def load_piper():
    global _piper_binary, _piper_model_path, _piper_config_path
    if _piper_binary is not None:
        return

    piper_bin = os.environ.get("PIPER_BIN", "piper")
    piper_path = shutil.which(piper_bin) if not os.path.isabs(piper_bin) else piper_bin
    if not piper_path or not os.path.exists(piper_path):
        raise RuntimeError(
            "Piper binary not found. Install Piper and set PIPER_BIN if needed."
        )

    model_path = os.environ.get(
        "PIPER_MODEL_PATH",
        os.path.join(os.getcwd(), "piper_models", "en_US-hfc_female-medium.onnx"),
    )
    config_path = os.environ.get("PIPER_CONFIG_PATH", f"{model_path}.json")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Piper model not found at {model_path}. Set PIPER_MODEL_PATH to en_US-hfc_female-medium.onnx."
        )
    if os.path.exists(config_path):
        _piper_config_path = config_path
    else:
        _piper_config_path = None

    _piper_binary = piper_path
    _piper_model_path = model_path


def synthesize_with_piper(text: str) -> tuple[str, str]:
    load_piper()
    temp_dir = os.path.join(os.getcwd(), "tmp_generate_audio")
    os.makedirs(temp_dir, exist_ok=True)
    output_wav = os.path.join(
        temp_dir,
        f"piper_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.wav",
    )

    cmd = [_piper_binary, "--model", _piper_model_path, "--output_file", output_wav]
    if _piper_config_path:
        cmd.extend(["--config", _piper_config_path])

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Piper synthesis failed: {stderr_text or 'unknown error'}")

    with open(output_wav, "rb") as f:
        wav_bytes = f.read()
    audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
    return audio_base64, output_wav


def _guess_audio_extension(mime_type: Optional[str]) -> str:
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/aac": "aac",
    }
    if not mime_type:
        return "wav"
    return mapping.get(mime_type.lower(), "bin")


def save_debug_input_audio(audio_bytes: bytes, mime_type: Optional[str] = None) -> str:
    temp_dir = os.path.join(os.getcwd(), "tmp_generate_audio")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    ext = _guess_audio_extension(mime_type)
    filename = f"input_{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    return filepath


def transcribe_audio_base64(audio_b64: str) -> tuple[str, Optional[str]]:
    try:
        import soundfile as sf
        import numpy as np
        import torchaudio.functional as F
    except Exception as e:
        raise RuntimeError(
            "Audio transcription requires soundfile and torchaudio in this environment."
        ) from e

    load_whisper()

    payload = audio_b64.strip()
    mime_type = None
    if "," in payload and payload.lower().startswith("data:"):
        header, payload = payload.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].strip().lower()

    try:
        audio_bytes = base64.b64decode(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail="input_audio_base64 is not valid base64.") from e

    saved_audio_path = None
    # try:
    #     saved_audio_path = save_debug_input_audio(audio_bytes, mime_type)
    #     print(f"[debug] Saved input audio: {saved_audio_path}")
    # except Exception as e:
    #     print(f"[debug] Failed to save input audio: {e}")

    try:
        audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode audio. Provide a valid audio file encoded as base64.",
        ) from e

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size > 0 else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak

    target_sr = int(getattr(_whisper_processor.feature_extractor, "sampling_rate", 16000))
    if sample_rate != target_sr:
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32)
        audio_np = F.resample(audio_tensor, sample_rate, target_sr).numpy()

    if audio_np.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty.")

    whisper_param = next(_whisper_model.parameters())
    whisper_device = whisper_param.device
    whisper_dtype = whisper_param.dtype
    whisper_inputs = _whisper_processor(
        audio=audio_np,
        sampling_rate=target_sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = whisper_inputs["input_features"].to(device=whisper_device, dtype=whisper_dtype)
    attention_mask = whisper_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=whisper_device)
    else:
        # Whisper may omit attention mask in some preprocessing paths.
        attention_mask = torch.ones(
            (input_features.shape[0], input_features.shape[-1]),
            device=whisper_device,
            dtype=torch.long,
        )

    with torch.inference_mode():
        predicted_ids = _whisper_model.generate(
            input_features,
            attention_mask=attention_mask,
            language="en",
            task="transcribe",
        )

    text = _whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    print(f"Transcription: {text}")
    
    if not text:
        raise HTTPException(status_code=400, detail="Transcription produced empty text.")
    return text, saved_audio_path


def runtime_info() -> dict:
    model_device = str(next(_llm_model.parameters()).device) if _llm_model is not None else "unloaded"
    model_dtype = str(next(_llm_model.parameters()).dtype) if _llm_model is not None else "unknown"
    return {
        "preferred_device": _device,
        "model_device": model_device,
        "model_dtype": model_dtype,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "whisper_device": str(next(_whisper_model.parameters()).device) if _whisper_model is not None else "unloaded",
    }


@app.on_event("startup")
async def startup_load_models():
    print("[startup] Loading models...")
    load_llm()
    print(f"[startup] Text model loaded with runtime: {runtime_info()}")


@app.post("/generate")
async def generate(req: GenerateRequest):
    message_text = (req.message or "").strip()
    has_audio_input = bool((req.input_audio_base64 or "").strip())

    if not message_text and not has_audio_input:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'message' (text) or 'input_audio_base64' (audio).",
        )

    # Ensure text model is loaded (startup preloads this).
    load_llm()

    transcribed_text = None
    saved_audio_path = None
    if not message_text and has_audio_input:
        transcribed_text, saved_audio_path = transcribe_audio_base64(req.input_audio_base64 or "")
        message_text = transcribed_text

    model_device = next(_llm_model.parameters()).device
    system_prompt = BASE_SYSTEM_PROMPT.strip()
    if req.system and req.system.strip() and req.system.strip() != "You are a helpful assistant.":
        system_prompt = f"{system_prompt}\n\nAdditional runtime instructions:\n{req.system.strip()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message_text},
    ]

    # Use model-native chat formatting when available.
    if hasattr(_llm_tokenizer, "apply_chat_template") and _llm_tokenizer.chat_template:
        inputs = _llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=model_device)
        prompt_len = int(inputs["input_ids"].shape[-1])
    else:
        prompt = chat_format(system_prompt, message_text)
        inputs = _llm_tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=model_device)
        prompt_len = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        # Prefer deterministic decoding for cleaner assistant-style replies.
        # Gemma chat models can emit multi-lingual/token-noise tails when sampling.
        eot_token_id = _llm_tokenizer.convert_tokens_to_ids("<end_of_turn>")
        stop_token_ids = [_llm_tokenizer.eos_token_id]
        if isinstance(eot_token_id, int) and eot_token_id >= 0:
            stop_token_ids.append(eot_token_id)

        max_new_tokens = req.max_new_tokens or 128
        max_new_tokens = max(1, min(int(max_new_tokens), 128))
        output_ids = _llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_llm_tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
        )

    generated_ids = output_ids[0][prompt_len:]
    generated_text = _llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    info = runtime_info()
    info["input_mode"] = "audio" if transcribed_text is not None else "text"
    if transcribed_text is not None:
        info["transcription"] = transcribed_text
        info["input_audio_file"] = saved_audio_path

    if not req.audio:
        return {"text": generated_text, "runtime": info}

    # TTS (Piper)
    try:
        audio_base64, output_wav_path = synthesize_with_piper(generated_text)
        info["tts_backend"] = "piper"
        info["tts_model"] = os.path.basename(_piper_model_path) if _piper_model_path else "unknown"
        info["tts_output_file"] = output_wav_path
        return {"text": generated_text, "audio": audio_base64, "mime": "audio/wav", "runtime": info}
    except Exception as e:
        info["tts_error"] = str(e)
        return JSONResponse({"text": generated_text, "audio_error": str(e), "runtime": info}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


