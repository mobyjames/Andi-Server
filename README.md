# Andi Python Server

A support server for "Andi" the Android (an Android device assistant)

Python FastAPI server with:
- Text generation: `google/gemma-3-1b-it`
- Speech-to-text: `openai/whisper-base`
- Text-to-speech: Piper (`en_US-hfc_female-medium`)

The server auto-selects accelerator priority:
1. CUDA GPU
2. Apple MPS
3. CPU

The server also auto-loads `.env` on startup.

## 1) Prerequisites

- Python `3.10` or `3.11`
- `pip`
- Optional but recommended: GPU (NVIDIA CUDA or Apple MPS)

If using VS Code, install the Python extension and select your project venv as interpreter.

## 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -V
```

## 3) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins tested package versions for reproducible setup.

## 4) Install Piper (macOS, pipx path used in this project)

Install Piper with `pipx`:

```bash
pipx install piper-tts
```

Verify:

```bash
pipx run piper --help
```

If needed, set `PIPER_BIN` to the full executable path (examples below).

## 5) Add Piper model files

Place these files in your project (default location):

- `piper_models/en_US-hfc_female-medium.onnx`
- `piper_models/en_US-hfc_female-medium.onnx.json`

You can also store them anywhere and set `PIPER_MODEL_PATH`/`PIPER_CONFIG_PATH`.

## 6) Configure `.env`

Example `.env`:

```env
# Required by /handshake route
HANDSHAKE_API_KEY=your_handshake_key

# Optional: only if your client calls /handshake and needs this value
PORCUPINE_ACCESS_KEY=your_porcupine_key

# Optional: Hugging Face auth for model download/rate limits
HF_TOKEN=hf_xxx

# Optional server port (default 3000)
PORT=3000

# Piper settings
# If piper is on PATH, this can be just "piper"
# For pipx installs, an explicit path is often safest:
PIPER_BIN=/Users/your-user/.local/bin/piper
PIPER_MODEL_PATH=/absolute/path/to/piper_models/en_US-hfc_female-medium.onnx
PIPER_CONFIG_PATH=/absolute/path/to/piper_models/en_US-hfc_female-medium.onnx.json

# Optional Whisper model override
WHISPER_MODEL_ID=openai/whisper-base
```

Notes:
- `.env` values are loaded automatically at app startup.
- Existing shell environment variables override `.env`.

## 7) Start the server

```bash
python server.py
```

Or:

```bash
uvicorn server:app --host 0.0.0.0 --port 3000
```

Health check:

```bash
curl http://127.0.0.1:3000/health
```

## 8) Test `/generate`

Text only:

```bash
curl -X POST http://127.0.0.1:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"message":"Write one short sentence about space.","max_new_tokens":32,"audio":false}'
```

Text + spoken output (Piper):

```bash
curl -X POST http://127.0.0.1:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"message":"Say hello in a friendly way.","max_new_tokens":32,"audio":true}'
```

Audio input (Whisper transcription -> text generation):

```bash
curl -X POST http://127.0.0.1:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_audio_base64":"<BASE64_AUDIO>","max_new_tokens":32,"audio":false}'
```

## 9) VS Code debug

1. Open folder in VS Code.
2. Select interpreter: `.venv/bin/python`.
3. Run and Debug -> create `launch.json` (Python), then use:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: server.py",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/server.py",
      "console": "integratedTerminal"
    }
  ]
}
```

## Troubleshooting

### `RuntimeError: Invalid buffer size: ... GiB`
- Model load is hitting memory limits.
- Lower `max_new_tokens` (e.g. 8-32), close other apps, and retry.

### `Piper binary not found`
- Confirm install:
  - `pipx run piper --help`
- Set `PIPER_BIN` to the real path, e.g. `/Users/<you>/.local/bin/piper`.

### Piper model/config not found
- Ensure both files exist and env vars point to correct absolute paths:
  - `.onnx`
  - `.onnx.json`

### Slow responses
- First run is slower due to model load/download.
- On constrained memory systems, even MPS may be slow; keep token counts low.
