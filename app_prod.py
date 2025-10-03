#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production NADRA Chatbot with FastAPI + Uvicorn
==============================================

Serves a static JS chat UI and provides:
- POST /api/transcribe  (multipart 'audio')  -> uses pre-loaded Whisper model
- POST /api/query       (json {"text": ...}) -> runs RAG routing
- Automatic OpenAPI docs at /docs

Model loads once at startup with device detection + warm-up test.
Run with: uvicorn app_prod:app --host 0.0.0.0 --port 5003
"""

import os
import sys
import tempfile
import warnings
import logging
import shutil
import subprocess
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

import ollama
import chromadb
from gtts import gTTS

from llm import tie_breaker_llm  # your existing tie-breaker using granite3.3:2b

# --- Quiet mode for model loading ---
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)
for noisy in ["urllib3", "numba", "matplotlib", "torchaudio", "transformers"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# --- Safe Torch import ---
try:
    import torch
except Exception:
    class _Shim:
        def __getattr__(self, _): return False
    torch = _Shim()  # type: ignore

# --------------------------
# Config
# --------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

DB_PATH = "chroma_bge_m3_db"
COLLECTION_NAME = "qa_collection"
EMBED_MODEL = "bge-m3:latest"

# ChromaDB uses inverted cosine distance (0.0 = perfect match)
T_HIGH = 0.18  # ≤ 0.18 = high confidence
T_LOW = 0.28   # ≤ 0.28 = medium confidence

TOPK_MEDIUM = 5
TOPK_FAST = 1

# Global model instance
whisper_model = None
current_device = None
collection = None

# --------------------------
# Pydantic Models
# --------------------------
class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    route: str
    llm: str
    score: Optional[float]

class TranscribeResponse(BaseModel):
    text: str

class ErrorResponse(BaseModel):
    error: str

# --------------------------
# Device Detection & Model Loading
# --------------------------
def get_device() -> str:
    """Device priority: CUDA -> MPS -> CPU"""
    try:
        if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)):
            if torch.cuda.is_available():
                return "cuda"
    except Exception:
        pass

    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
    except Exception:
        pass

    return "cpu"

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def convert_to_wav_16k_mono(input_path: str) -> str:
    """Convert any audio file to 16kHz, mono WAV."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    # Try ffmpeg first
    if has_ffmpeg():
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", input_path,
                    "-ac", "1",
                    "-ar", "16000",
                    "-f", "wav",
                    "-sample_fmt", "s16",
                    tmp_path,
                ],
                check=True,
            )
            return tmp_path
        except subprocess.CalledProcessError:
            pass

    # Fallback: librosa + soundfile
    try:
        import librosa
        import soundfile as sf
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(tmp_path, audio, 16000, subtype="PCM_16", format="WAV")
        return tmp_path
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"Audio conversion failed. Install FFmpeg or librosa+soundfile. Details: {e}")

def load_whisper_model():
    """Load appropriate Whisper model based on device"""
    global whisper_model, current_device
    
    current_device = get_device()
    print(f"🔧 Detected device: {current_device.upper()}")
    print(f"📥 Loading Whisper model...")
    
    try:
        if current_device == "cuda":
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("large-v2", device="cuda", compute_type="float16")
            print("✅ CUDA Whisper model loaded (large-v2, float16)")
            
        elif current_device == "mps":
            # For MPS, we'll use mlx_whisper - model loads on first transcribe call
            import mlx_whisper
            whisper_model = "mlx_whisper_loaded"  # Flag that MLX is ready
            print("✅ MPS MLX Whisper ready (large-v2, 8-bit)")
            
        else:  # CPU
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("large-v2", device="cpu", compute_type="int8")
            print("✅ CPU Whisper model loaded (large-v2, int8)")
            
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        raise

def transcribe_audio(input_path: str, language: str = "en") -> str:
    """Transcribe audio using pre-loaded model"""
    global whisper_model, current_device
    
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    
    try:
        if current_device == "cuda" or current_device == "cpu":
            # Use faster-whisper
            segments, _info = whisper_model.transcribe(
                input_path,
                beam_size=5,
                language=language
            )
            return " ".join(seg.text for seg in segments)
            
        elif current_device == "mps":
            # Use MLX Whisper - convert to WAV first for reliability
            wav_path = convert_to_wav_16k_mono(input_path)
            try:
                import mlx_whisper
                result = mlx_whisper.transcribe(
                    wav_path,
                    path_or_hf_repo="mlx-community/whisper-large-v2-mlx-8bit",
                    language=language,
                    word_timestamps=False
                )
                return (result.get("text") or "").strip()
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        
        return ""
        
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

def generate_warmup_audio():
    """Generate Hello.mp3 using gTTS if it doesn't exist"""
    hello_path = os.path.join(os.path.dirname(__file__), "Hello.mp3")
    
    if os.path.exists(hello_path):
        print(f"📁 Found existing Hello.mp3, skipping generation")
        return hello_path
    
    print(f"🎵 Generating Hello.mp3 using gTTS...")
    try:
        tts = gTTS(text="Hello! Transcription Model Loaded", lang='en', slow=False)
        tts.save(hello_path)
        print(f"✅ Hello.mp3 generated successfully")
        return hello_path
    except Exception as e:
        print(f"⚠️  Failed to generate Hello.mp3: {e}")
        return None

def warmup_model():
    """Warm up the model by transcribing Hello.mp3"""
    hello_path = generate_warmup_audio()
    
    if hello_path and os.path.exists(hello_path):
        print(f"🔥 Warming up model with Hello.mp3...")
        try:
            result = transcribe_audio(hello_path)
            print(f"🎯 Warmup result: '{result.strip()}'")
            print(f"✅ Model warmed up successfully!")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
    else:
        print(f"⚠️  Skipping warmup - no Hello.mp3 available")

def load_chromadb():
    """Initialize ChromaDB connection"""
    global collection
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print("✅ ChromaDB collection loaded")
    except Exception as e:
        print(f"❌ Failed to load ChromaDB: {e}")
        raise

# --------------------------
# App Lifecycle Management
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 60)
    print("🚀 Starting NADRA Chatbot with FastAPI + Uvicorn")
    print("=" * 60)
    
    try:
        # Load models and databases
        load_whisper_model()
        load_chromadb()
        warmup_model()
        
        print("=" * 60)
        print("🎉 Ready to serve!")
        print("=" * 60)
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(
    title="NADRA Chatbot API",
    description="Production chatbot with Whisper transcription and RAG-based Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# --------------------------
# Embeddings & Utilities (unchanged)
# --------------------------
def _embed(text: str) -> List[float]:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

def _get_distance(results, index: int = 0) -> float:
    """Extract distance from ChromaDB results"""
    try:
        if 'distances' in results and results['distances'] and len(results['distances'][0]) > index:
            return float(results['distances'][0][index])
        return 0.0
    except Exception:
        return 0.0

def _get_field(results, index: int, key: str, default=""):
    """Extract field from ChromaDB results metadata"""
    try:
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas'][0]) > index:
            metadata = results['metadatas'][0][index]
            if metadata and isinstance(metadata, dict):
                return metadata.get(key, default)
        return default
    except Exception:
        return default

def _get_document(results, index: int = 0, default=""):
    """Extract document (question) from ChromaDB results"""
    try:
        if 'documents' in results and results['documents'] and len(results['documents'][0]) > index:
            return results['documents'][0][index] or default
        return default
    except Exception:
        return default

def _get_id(results, index: int = 0, default=""):
    """Extract ID from ChromaDB results"""
    try:
        if 'ids' in results and results['ids'] and len(results['ids'][0]) > index:
            return results['ids'][0][index] or default
        return default
    except Exception:
        return default

# --------------------------
# Replies (unchanged)
# --------------------------
def _default_message() -> str:
    return (
        "I'm focused on NADRA information—centers, fees, required documents, processing times, and general guidance.\n"
        "I couldn't find anything relevant to your query.\n\n"
        "For center details (addresses, timings, phone numbers): https://www.nadra.gov.pk/nadraOffices\n"
        "For international offices: https://www.nadra.gov.pk/internationalOffices\n"
        "For assistance, call NADRA helpline: 1777 (from mobile) or +92 51 111 786 100."
    )

def _location_general_answer() -> str:
    return (
        "NADRA operates multiple centers, including some extended/24/7 facilities; availability and timings vary by branch.\n"
        "For addresses, timings, phone numbers and maps, please check the official directories:\n"
        "• Domestic centers: https://www.nadra.gov.pk/nadraOffices\n"
        "• International offices: https://www.nadra.gov.pk/internationalOffices\n\n"
        "Tip: Use the listed phone number to confirm same-day hours or facilities (e.g., executive/ladies/wheelchair access).\n"
        "You can also call the NADRA helpline at 1777 (from mobile) or +92 51 111 786 100."
    )

# --------------------------
# Rule-based location intent (unchanged)
# --------------------------
_LOCATION_KEYWORDS = [
    "office", "center", "centre", "branch", "location", "address", "map", "near me",
    "timings", "time", "hours", "open", "close", "sunday", "saturday", "weekend",
    "helpline", "phone", "number", "contact", "mrv", "mobile van", "kiosk",
    "executive", "ladies", "female", "wheelchair", "accessible", "accessibility",
    "queue", "token", "appointment", "ramzan", "ramadan",
    "khula", "band", "kidhar", "kahan", "kidar", "qareeb", "idhar", "udhar", "nazdeek"
]
_NADRA_CUES = ["nadra", "cnic", "nicop", "poc", "frc", "crc"]

def _contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def is_location_intent(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    has_loc = _contains_any(q, _LOCATION_KEYWORDS)
    has_nadra = _contains_any(q, _NADRA_CUES)
    return bool(has_loc and has_nadra)

# --------------------------
# Core routing (unchanged)
# --------------------------
def answer_query(user_query: str) -> dict:
    uq = (user_query or "").strip()
    if not uq:
        return {"answer": "Please enter a question.", "route": "N/A", "llm": "N/A", "score": None}

    try:
        vec = _embed(uq)
        results = collection.query(
            query_embeddings=[vec],
            n_results=max(TOPK_FAST, TOPK_MEDIUM),
            include=['metadatas', 'distances', 'documents']
        )

        if not results or not results.get('distances') or not results['distances'][0]:
            if is_location_intent(uq):
                print(f'User query: "{uq}", Classified: T_Low, LLM result: N/A')
                return {"answer": _location_general_answer(), "route": "T_Low", "llm": "N/A", "score": None}
            print(f'User query: "{uq}", Classified: T_Low, LLM result: N/A')
            return {"answer": _default_message(), "route": "T_Low", "llm": "N/A", "score": None}

        top_score = _get_distance(results, 0)
        top_answer = _get_field(results, 0, "answer", "")
        
        print(f'[DEBUG] Query: "{uq[:50]}..." | Distance: {top_score:.4f}', file=sys.stderr)

        # T_high: distance <= 0.18
        if top_score <= T_HIGH:
            print(f'User query: "{uq}", Classified: T_High, LLM result: N/A')
            return {"answer": top_answer or "No answer found.", "route": "T_High", "llm": "N/A", "score": top_score}

        # T_medium: distance <= 0.28 but > 0.18
        if top_score <= T_LOW:
            candidates, candidate_ids = [], []
            num_results = min(TOPK_MEDIUM, len(results['distances'][0]))
            
            for i in range(num_results):
                question = _get_document(results, i, "")
                candidates.append(question)
                result_id = _get_id(results, i, str(i + 1))
                candidate_ids.append(result_id)

            while len(candidates) < TOPK_MEDIUM:
                candidates.append("N/A")
                candidate_ids.append("0")

            try:
                tb = tie_breaker_llm(uq, candidates, candidate_ids)
                if tb.get("match") == "YES":
                    best_id = str(tb["best_id"])
                    best_idx = 0
                    try:
                        best_idx = int(best_id) - 1
                        if best_idx < 0 or best_idx >= num_results:
                            best_idx = 0
                    except ValueError:
                        for idx in range(num_results):
                            if _get_id(results, idx) == best_id:
                                best_idx = idx
                                break
                    
                    ans = _get_field(results, best_idx, "answer", "") or "No answer found."
                    print(f'User query: "{uq}", Classified: T_Medium, LLM result: YES best_id={tb["best_id"]}')
                    return {"answer": ans, "route": "T_Medium", "llm": f'YES best_id={tb["best_id"]}', "score": top_score}
                else:
                    print(f'User query: "{uq}", Classified: T_Medium, LLM result: NO')
                    return {"answer": _default_message(), "route": "T_Medium", "llm": "NO", "score": top_score}
            except Exception as e:
                print(f'User query: "{uq}", Classified: T_Medium, LLM result: ERROR ({e})')
                return {"answer": _default_message(), "route": "T_Medium", "llm": f'ERROR ({e})', "score": top_score}

        # T_low: distance > 0.28
        if is_location_intent(uq):
            print(f'User query: "{uq}", Classified: T_Low, LLM result: N/A')
            return {"answer": _location_general_answer(), "route": "T_Low", "llm": "N/A", "score": top_score}

        print(f'User query: "{uq}", Classified: T_Low, LLM result: N/A')
        return {"answer": _default_message(), "route": "T_Low", "llm": "N/A", "score": top_score}

    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return {"answer": f"Search error: {e}", "route": "ERROR", "llm": "N/A", "score": None}

# --------------------------
# API Routes
# --------------------------
@app.get("/")
async def root():
    """Serve the main frontend page"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Please ensure frontend directory exists."}

# Serve individual static files at root level (matching Flask behavior)
@app.get("/{file_path:path}")
async def serve_static_files(file_path: str):
    """Serve static files from frontend directory"""
    # Only serve files that exist in frontend directory
    full_path = os.path.join(FRONTEND_DIR, file_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    # If file not found, return 404
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """Process text query using RAG pipeline"""
    try:
        result = answer_query(request.text)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def api_transcribe(audio: UploadFile = File(...)):
    """
    Transcribe audio using pre-loaded Whisper model.
    Accepts multipart file upload.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file to temporary location
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use pre-loaded model for transcription
        text = transcribe_audio(tmp_path)
        return TranscribeResponse(text=text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "device": current_device,
        "chromadb_loaded": collection is not None
    }

# --------------------------
# Static File Serving (like original Flask setup)  
# --------------------------
# Remove the manual root route and catch-all - let StaticFiles handle everything
# API routes defined above take precedence over static files

# --------------------------
# Run with Uvicorn
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5003"))
    uvicorn.run(
        "app_prod:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )