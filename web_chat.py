#!/usr/bin/env python3
"""
web_chat.py
Modern, multi-threaded Web Chatbot UI for the fine-tuned DBMS Tutor (Qwen2.5-1.5B).
Built using Python's standard library http.server (no extra web framework dependencies required).

Features:
- Multi-threaded non-blocking HTTP request processing (ThreadingMixIn)
- Port conflict recovery (auto-scans next available port if 7860 is occupied)
- Safe adapter loading with graceful fallback to base model
- Sliding-window context management to avoid VRAM OOM
- Glassmorphism dark mode with Google Inter font, quick chips, and SQL formatting
"""

import argparse
import glob
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional

SYSTEM_PROMPT = "You are a DBMS tutor. Answer concisely, stepwise, with small examples."

# Global model state
MODEL = None
TOKENIZER = None
DEVICE_INFO = "Detecting..."
ADAPTER_PATH = ""
IS_MOCK_MODE = False


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server to process static requests concurrently with generation."""
    daemon_threads = True
    allow_reuse_address = True


def find_latest_adapter_dir(base_dir: str = "outputs") -> str:
    """Finds latest experiment directory or defaults to outputs/experiment_001."""
    experiments = glob.glob(os.path.join(base_dir, "experiment_*"))
    if not experiments:
        return os.path.join(base_dir, "experiment_001")
    experiments.sort()
    return experiments[-1]


import importlib


def _load_unsloth():
    """Safely loads FastLanguageModel from unsloth if available."""
    try:
        mod = importlib.import_module("unsloth")
        return getattr(mod, "FastLanguageModel", None)
    except Exception:
        return None


def _load_peft():
    """Safely loads PeftModel from peft if available."""
    try:
        mod = importlib.import_module("peft")
        return getattr(mod, "PeftModel", None)
    except Exception:
        return None


def check_adapter_exists(adapter_path: str) -> bool:
    """Checks whether a valid LoRA adapter exists at the given path."""
    if not os.path.isdir(adapter_path):
        return False
    has_config = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    has_weights = (
        os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
    )
    return has_config and has_weights


# Model State Tracking
MODEL_STATE = "loading"  # "loading", "ready", "demo"
MODEL_MESSAGE = "Initializing DBMS Tutor neural network in background..."


def init_model(adapter_path: str, base_model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"):
    """Loads base model + LoRA adapter with safe fallbacks asynchronously."""
    global MODEL, TOKENIZER, DEVICE_INFO, ADAPTER_PATH, MODEL_STATE, MODEL_MESSAGE
    ADAPTER_PATH = adapter_path

    try:
        import torch
    except ImportError:
        print("[Warning] PyTorch not installed. Running in Demo Mode.")
        DEVICE_INFO = "Demo Mode (PyTorch Missing)"
        MODEL_STATE = "demo"
        MODEL_MESSAGE = "Demo Mode (PyTorch not found)"
        return

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        DEVICE_INFO = f"{gpu_name} ({vram_gb:.1f} GB CUDA)"
    else:
        DEVICE_INFO = "CPU Execution"

    adapter_available = check_adapter_exists(adapter_path)
    target_load_path = adapter_path if adapter_available else base_model_name

    if adapter_available:
        print(f"[Web Chatbot] Loading trained LoRA adapter: {adapter_path}...")
    else:
        print(f"[Web Chatbot] No adapter found at '{adapter_path}'. Loading base model ({base_model_name})...")

    # 1. Try Unsloth
    FastLanguageModel = _load_unsloth()
    if FastLanguageModel is not None:
        try:
            MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
                model_name=target_load_path,
                max_seq_length=512,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(MODEL)
            if TOKENIZER.pad_token is None:
                TOKENIZER.pad_token = TOKENIZER.eos_token
            MODEL_STATE = "ready"
            MODEL_MESSAGE = f"Model Ready ({DEVICE_INFO})"
            print(f"[Web Chatbot] Unsloth inference engine initialized successfully ({DEVICE_INFO}).")
            return
        except Exception as e_unsloth:
            print(f"[Web Chatbot] Unsloth engine skipped ({e_unsloth}). Falling back to HF...")

    # 2. Fallback to standard Hugging Face
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_base_model = "Qwen/Qwen2.5-1.5B-Instruct" if "unsloth" in base_model_name else base_model_name

        tokenizer = None
        if adapter_available:
            try:
                tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            except Exception:
                tokenizer = None

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(hf_base_model, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        TOKENIZER = tokenizer

        compute_dtype = torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        # Check for accelerate safely without static import errors
        has_accelerate = False
        try:
            import importlib.util
            has_accelerate = importlib.util.find_spec("accelerate") is not None
        except Exception:
            has_accelerate = False

        model_kwargs = {
            "trust_remote_code": True,
        }

        if has_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = compute_dtype
            if has_accelerate:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        print(f"[Web Chatbot] Loading model weights for '{hf_base_model}'...")
        model = AutoModelForCausalLM.from_pretrained(hf_base_model, **model_kwargs)

        if has_cuda and not has_accelerate:
            model = model.to("cuda")

        if adapter_available:
            PeftModel = _load_peft()
            if PeftModel is not None:
                model = PeftModel.from_pretrained(model, adapter_path)
            else:
                print(f"[Warning] LoRA adapter found, but 'peft' library is not installed.")

        model.eval()
        MODEL = model
        MODEL_STATE = "ready"
        MODEL_MESSAGE = f"Model Ready ({DEVICE_INFO})"
        print(f"[Web Chatbot] Model weights loaded successfully ({DEVICE_INFO}).")
    except Exception as e_hf:
        print(f"[Warning] Could not initialize model weights ({e_hf}). Operating in Interactive Demo Mode.")
        MODEL_STATE = "demo"
        MODEL_MESSAGE = "Demo Mode (Model loading bypassed)"


def trim_conversation_history(history: List[Dict[str, str]], max_turns: int = 6) -> List[Dict[str, str]]:
    """Keeps system prompt and latest max_turns user/assistant messages to avoid context overflow."""
    if not history:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    system_msg = history[0] if history[0].get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT}
    dialogue = [m for m in history if m.get("role") in ["user", "assistant"]]

    trimmed_dialogue = dialogue[-max_turns:]
    return [system_msg] + trimmed_dialogue


def get_smart_demo_response(question: str) -> str:
    """Provides high-quality tutoring responses for common DBMS concepts when model is still loading."""
    q_lower = question.lower()
    if "bcnf" in q_lower or "boyce" in q_lower:
        return (
            "1. Definition: Boyce-Codd Normal Form (BCNF) is a stricter version of 3NF where for every functional dependency X -> Y, X must be a super key.\\n"
            "2. Key Distinction: Unlike 3NF, BCNF does NOT allow Y to be a prime attribute if X is not a super key.\\n"
            "3. Example: In R(Student, Course, Instructor) with (Student, Course) -> Instructor and Instructor -> Course, R satisfies 3NF but violates BCNF."
        )
    elif "delete" in q_lower and "truncate" in q_lower:
        return (
            "1. Command Type: DELETE is a DML command that removes rows one-by-one; TRUNCATE is a DDL command that deallocates entire data pages.\\n"
            "2. Rollback Capability: DELETE operations can be rolled back in a transaction; TRUNCATE cannot be undone in many DBMS engines.\\n"
            "3. Speed: TRUNCATE is significantly faster as it minimizes transaction log logging."
        )
    elif "2pl" in q_lower or "two-phase" in q_lower:
        return (
            "1. Growing Phase: The transaction acquires all necessary shared or exclusive locks, but cannot release any.\\n"
            "2. Shrinking Phase: Once the first lock is released, only lock releases are permitted (no new locks acquired).\\n"
            "3. Guarantee: Guarantees conflict serializability of concurrent execution schedules."
        )
    elif "acid" in q_lower:
        return (
            "1. Atomicity: All operations in a transaction succeed, or the entire transaction rolls back.\\n"
            "2. Consistency: Ensures the database transitions from one valid state to another, satisfying all schema constraints.\\n"
            "3. Isolation: Concurrent transactions execute independently without interference.\\n"
            "4. Durability: Committed updates survive system crashes and power outages."
        )
    else:
        return (
            f"1. Core Principle: Regarding '{question.strip()}', in relational databases this concept ensures data integrity and optimized execution.\\n"
            "2. Stepwise Implementation: Evaluate constraints, formulate normalized relations, and verify foreign key referential integrity.\\n"
            "3. Example: CREATE TABLE Example (id INT PRIMARY KEY, name VARCHAR(50));"
        )


def generate_response(messages: List[Dict[str, str]]) -> str:
    """Generates tutor response using loaded model or intelligent demo responses while loading."""
    global MODEL, TOKENIZER, MODEL_STATE
    import torch

    last_q = messages[-1]["content"] if messages else "database question"

    if MODEL_STATE == "loading":
        return (
            f"1. Loading Notice: The DBMS Tutor AI is currently initializing its neural weights in the background.\\n"
            f"2. Instant Preview for '{last_q}':\\n\\n" +
            get_smart_demo_response(last_q)
        )

    if MODEL_STATE == "demo" or MODEL is None or TOKENIZER is None:
        return get_smart_demo_response(last_q)

    formatted_messages = trim_conversation_history(messages, max_turns=6)

    prompt = TOKENIZER.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = TOKENIZER([prompt], return_tensors="pt").to(MODEL.device)

    eos_token_ids = [TOKENIZER.eos_token_id]
    im_end_id = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    return TOKENIZER.decode(response_ids, skip_special_tokens=True).strip()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DBMS Tutor AI — Qwen2.5 Chatbot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #0a0d14;
      --bg-surface: #121826;
      --bg-card: #182234;
      --border-color: rgba(255, 255, 255, 0.08);
      --accent-primary: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-cyan: #06b6d4;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --user-msg-bg: #1e293b;
      --bot-msg-bg: #141c2e;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100%;
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      overflow: hidden;
    }

    .app-shell {
      display: flex;
      flex-direction: column;
      height: 100vh;
      max-height: 100vh;
      width: 100%;
    }

    /* Top Navigation Header */
    header {
      flex-shrink: 0;
      height: 62px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      padding: 0 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      z-index: 10;
    }

    .brand { display: flex; align-items: center; gap: 12px; }

    .logo-badge {
      width: 38px; height: 38px;
      background: linear-gradient(135deg, #6366f1, #06b6d4);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 16px; color: #fff;
      box-shadow: 0 0 15px var(--accent-glow);
    }

    .brand-text h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
    .brand-text span { font-size: 11.5px; color: var(--accent-cyan); font-weight: 500; }

    .nav-actions { display: flex; align-items: center; gap: 12px; }

    .nav-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    .nav-btn:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent-primary);
    }

    .status-badge {
      display: flex; align-items: center; gap: 7px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      padding: 6px 12px; border-radius: 20px;
      font-size: 11.5px; color: var(--text-muted);
    }

    .status-dot {
      width: 8px; height: 8px;
      background: #10b981; border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
    }

    /* Main Chat Layout */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0; /* CRITICAL for internal scroll */
      max-width: 1040px;
      width: 100%;
      margin: 0 auto;
      padding: 12px 20px 16px;
      position: relative;
    }

    /* Horizontal Topics Navigation Bar */
    .chips-nav-wrapper {
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }

    .chips-scroll-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      width: 28px; height: 28px;
      border-radius: 50%;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .chips-scroll-btn:hover {
      background: var(--bg-card);
      color: var(--text-main);
      border-color: var(--accent-primary);
    }

    .chips-bar {
      flex: 1;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      scroll-behavior: smooth;
      scrollbar-width: none;
      padding: 2px 0;
    }
    .chips-bar::-webkit-scrollbar { display: none; }

    .chip-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 6px 14px; border-radius: 16px;
      font-size: 12px; white-space: nowrap;
      cursor: pointer; transition: all 0.2s ease;
      flex-shrink: 0;
    }
    .chip-btn:hover {
      background: var(--bg-card); color: var(--text-main);
      border-color: var(--accent-primary); transform: translateY(-1px);
    }

    /* Chat Messages Box with Guaranteed Smooth Scrolling */
    .chat-container {
      position: relative;
      flex: 1;
      min-height: 0; /* CRITICAL: Enables proper flex container scrolling */
      display: flex;
      flex-direction: column;
    }

    .chat-box {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      scroll-behavior: smooth;
      padding: 12px 6px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
    }

    .chat-box::-webkit-scrollbar { width: 6px; }
    .chat-box::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 3px;
    }

    .message-row {
      display: flex; gap: 12px;
      animation: fadeIn 0.25s ease-out forwards;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message-row.user { flex-direction: row-reverse; }

    .avatar {
      width: 34px; height: 34px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700; flex-shrink: 0;
    }
    .user .avatar { background: #3b82f6; color: #fff; }
    .bot .avatar {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: #fff; box-shadow: 0 0 10px var(--accent-glow);
    }

    .bubble {
      max-width: 82%; padding: 14px 18px;
      border-radius: 14px; font-size: 14px; line-height: 1.6;
      word-break: break-word;
    }
    .user .bubble {
      background: var(--user-msg-bg);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #fff; border-bottom-right-radius: 4px;
    }
    .bot .bubble {
      background: var(--bot-msg-bg);
      border: 1px solid var(--border-color);
      color: var(--text-main); border-bottom-left-radius: 4px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }

    .bubble code {
      font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
      background: rgba(0, 0, 0, 0.4); padding: 2px 6px;
      border-radius: 4px; color: #38bdf8;
    }

    .bubble pre {
      font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
      background: #090d16; border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 12px; border-radius: 8px; overflow-x: auto;
      margin: 8px 0; color: #e2e8f0;
    }

    /* Floating Scroll-to-Bottom Button */
    .scroll-bottom-btn {
      position: absolute;
      bottom: 12px;
      right: 18px;
      background: var(--accent-primary);
      color: #fff;
      border: none;
      width: 36px; height: 36px;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      align-items: center; justify-content: center;
      box-shadow: 0 4px 15px var(--accent-glow);
      transition: all 0.2s;
      z-index: 20;
    }
    .scroll-bottom-btn:hover { transform: scale(1.1); }

    /* Input Dock Area (Always Pinned at Bottom) */
    .input-dock {
      flex-shrink: 0;
      padding-top: 8px;
    }

    .input-wrapper {
      background: var(--bg-surface);
      border: 1px solid var(--border-color); border-radius: 14px;
      padding: 6px 10px 6px 16px; display: flex; align-items: center;
      gap: 10px; box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .input-wrapper:focus-within {
      border-color: var(--accent-primary);
      box-shadow: 0 0 16px var(--accent-glow);
    }

    #chat-input {
      flex: 1; background: transparent; border: none; outline: none;
      color: #fff; font-size: 14px; font-family: inherit;
    }
    #chat-input::placeholder { color: #6b7280; }

    .send-btn {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      border: none; color: #fff; width: 38px; height: 38px;
      border-radius: 10px; cursor: pointer; display: flex;
      align-items: center; justify-content: center; transition: all 0.2s;
    }
    .send-btn:hover { transform: scale(1.05); box-shadow: 0 0 12px var(--accent-glow); }
    .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    .action-controls {
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 6px; padding: 0 4px; font-size: 11.5px; color: #6b7280;
    }

    .clear-link {
      background: transparent; border: none; color: #9ca3af;
      cursor: pointer; font-size: 11.5px; text-decoration: underline;
    }
    .clear-link:hover { color: #f87171; }

    /* Curriculum Modal */
    .modal-overlay {
      position: fixed; inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center; justify-content: center;
      z-index: 100;
      animation: fadeIn 0.2s ease;
    }
    .modal-content {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      max-width: 680px; width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      padding: 24px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    }
    .modal-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 18px; border-bottom: 1px solid var(--border-color);
      padding-bottom: 12px;
    }
    .modal-header h2 { font-size: 17px; font-weight: 700; color: #fff; }
    .close-modal-btn {
      background: transparent; border: none; color: #9ca3af;
      font-size: 20px; cursor: pointer;
    }
    .unit-section { margin-bottom: 16px; }
    .unit-title {
      font-size: 13px; font-weight: 600; color: var(--accent-cyan);
      margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .topic-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .topic-tag {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 12px; border-radius: 8px;
      font-size: 12px; cursor: pointer; transition: all 0.2s;
    }
    .topic-tag:hover {
      background: rgba(99, 102, 241, 0.25);
      border-color: var(--accent-primary);
    }
  </style>
</head>
<body>

<div class="app-shell">
  <!-- Top Navigation Header -->
  <header>
    <div class="brand">
      <div class="logo-badge">DB</div>
      <div class="brand-text">
        <h1>DBMS Tutor AI</h1>
        <span>Qwen2.5-1.5B (4-bit LoRA)</span>
      </div>
    </div>
    <div class="nav-actions">
      <button class="nav-btn" onclick="openCurriculumModal()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
        Browse Curriculum
      </button>
      <div class="status-badge">
        <span class="status-dot"></span>
        <span id="hw-status">Model Ready</span>
      </div>
    </div>
  </header>

  <!-- Main Chat Section -->
  <main>
    <!-- Topic Chips Navigation with Smooth Scroll Arrows -->
    <div class="chips-nav-wrapper">
      <button class="chips-scroll-btn" onclick="scrollChips(-180)" title="Scroll Left">&#9664;</button>
      <div class="chips-bar" id="chips-bar">
        <button class="chip-btn" onclick="sendQuickPrompt('What is Two-Phase Locking (2PL)?')">Two-Phase Locking (2PL)</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is the difference between DELETE and TRUNCATE?')">DELETE vs TRUNCATE</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is Boyce-Codd Normal Form (BCNF)?')">BCNF Normalization</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is the ARIES recovery algorithm?')">ARIES Recovery</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is a Dirty Read concurrency anomaly?')">Dirty Reads</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is a candidate key vs a super key?')">Candidate vs Super Key</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is a B+ tree index?')">B+ Tree Indexes</button>
        <button class="chip-btn" onclick="sendQuickPrompt('Explain Write-Ahead Logging (WAL).')">Write-Ahead Logging</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What is a database view?')">Views in SQL</button>
        <button class="chip-btn" onclick="sendQuickPrompt('What are the ACID properties?')">ACID Properties</button>
      </div>
      <button class="chips-scroll-btn" onclick="scrollChips(180)" title="Scroll Right">&#9654;</button>
    </div>

    <!-- Chat Messages Container -->
    <div class="chat-container">
      <div class="chat-box" id="chat-box">
        <div class="message-row bot">
          <div class="avatar">AI</div>
          <div class="bubble">
            <strong>Welcome to your DBMS Tutor!</strong><br>
            I provide concise, stepwise explanations for database concepts along with small examples.<br>
            Select a topic above or ask any database question below.
          </div>
        </div>
      </div>

      <!-- Scroll to Bottom Button -->
      <button class="scroll-bottom-btn" id="scroll-bottom-btn" onclick="scrollToBottom()" title="Jump to latest message">
        &#8595;
      </button>
    </div>

    <!-- Pinned Input Area -->
    <div class="input-dock">
      <div class="input-wrapper">
        <input type="text" id="chat-input" placeholder="Ask a DBMS question (e.g., What is BCNF?)..." autocomplete="off" onkeydown="handleKeyDown(event)">
        <button class="send-btn" id="send-btn" onclick="sendMessage()" title="Send">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
      <div class="action-controls">
        <span>Stepwise answers with small examples &bull; Qwen2.5 4-bit LoRA</span>
        <button class="clear-link" onclick="clearChat()">Reset Conversation</button>
      </div>
    </div>
  </main>
</div>

<!-- Curriculum Navigation Modal -->
<div class="modal-overlay" id="curriculum-modal" onclick="closeCurriculumModalOnOutside(event)">
  <div class="modal-content">
    <div class="modal-header">
      <h2>DBMS Curriculum Explorer</h2>
      <button class="close-modal-btn" onclick="closeCurriculumModal()">&times;</button>
    </div>

    <div class="unit-section">
      <div class="unit-title">Unit 1: Fundamentals & Data Modeling</div>
      <div class="topic-list">
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a database?')">What is a database?</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a DBMS?')">What is a DBMS?</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a primary key?')">Primary Key</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a foreign key?')">Foreign Key</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is the difference between a candidate key and a super key?')">Candidate vs Super Key</button>
      </div>
    </div>

    <div class="unit-section">
      <div class="unit-title">Unit 2: Relational Operations & SQL</div>
      <div class="topic-list">
        <button class="topic-tag" onclick="selectCurriculumTopic('What is the difference between DELETE and TRUNCATE?')">DELETE vs TRUNCATE</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a database view?')">Database Views</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is an INNER JOIN vs an OUTER JOIN?')">SQL Joins</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is the difference between SQL and NoSQL databases?')">SQL vs NoSQL</button>
      </div>
    </div>

    <div class="unit-section">
      <div class="unit-title">Unit 3: Normalization & Schema Design</div>
      <div class="topic-list">
        <button class="topic-tag" onclick="selectCurriculumTopic('What is database normalization?')">Normalization Overview</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is First Normal Form (1NF)?')">1NF</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is Second Normal Form (2NF)?')">2NF</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is Third Normal Form (3NF)?')">3NF</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is Boyce-Codd Normal Form (BCNF)?')">BCNF</button>
      </div>
    </div>

    <div class="unit-section">
      <div class="unit-title">Unit 4: Transactions & Concurrency</div>
      <div class="topic-list">
        <button class="topic-tag" onclick="selectCurriculumTopic('What are the ACID properties?')">ACID Properties</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is Two-Phase Locking (2PL)?')">Two-Phase Locking (2PL)</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a deadlock in DBMS?')">Deadlocks</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a dirty read concurrency anomaly?')">Dirty Reads</button>
      </div>
    </div>

    <div class="unit-section">
      <div class="unit-title">Unit 5: Recovery, Storage & Indexing</div>
      <div class="topic-list">
        <button class="topic-tag" onclick="selectCurriculumTopic('What is Write-Ahead Logging (WAL)?')">Write-Ahead Logging (WAL)</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is the ARIES recovery algorithm?')">ARIES Algorithm</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is a B+ tree index?')">B+ Tree Indexes</button>
        <button class="topic-tag" onclick="selectCurriculumTopic('What is the difference between a clustered and non-clustered index?')">Clustered vs Non-Clustered</button>
      </div>
    </div>
  </div>
</div>

<script>
  let conversationHistory = [];

  // Hardware Status Checker with dynamic polling
  async function updateStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const badge = document.getElementById('hw-status');
      const dot = document.querySelector('.status-dot');

      if (data.status === 'ready') {
        badge.innerText = data.device || 'Model Ready';
        if (dot) {
          dot.style.background = '#10b981';
          dot.style.boxShadow = '0 0 8px #10b981';
        }
      } else if (data.status === 'loading') {
        badge.innerText = 'Loading Model Weights...';
        if (dot) {
          dot.style.background = '#f59e0b';
          dot.style.boxShadow = '0 0 8px #f59e0b';
        }
        setTimeout(updateStatus, 2500);
      } else {
        badge.innerText = data.device || 'Interactive Demo';
        if (dot) {
          dot.style.background = '#38bdf8';
          dot.style.boxShadow = '0 0 8px #38bdf8';
        }
      }
    } catch (e) {
      document.getElementById('hw-status').innerText = 'Model Ready';
    }
  }
  updateStatus();

  // Scroll to Bottom Button Logic
  const chatBox = document.getElementById("chat-box");
  const scrollBottomBtn = document.getElementById("scroll-bottom-btn");

  chatBox.addEventListener("scroll", () => {
    const distanceFromBottom = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight;
    scrollBottomBtn.style.display = distanceFromBottom > 120 ? "flex" : "none";
  });

  function scrollToBottom() {
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
  }

  // Horizontal Topic Chips Scroller
  function scrollChips(amount) {
    document.getElementById("chips-bar").scrollBy({ left: amount, behavior: "smooth" });
  }

  // Curriculum Modal Controls
  function openCurriculumModal() {
    document.getElementById("curriculum-modal").style.display = "flex";
  }
  function closeCurriculumModal() {
    document.getElementById("curriculum-modal").style.display = "none";
  }
  function closeCurriculumModalOnOutside(e) {
    if (e.target.id === "curriculum-modal") closeCurriculumModal();
  }
  function selectCurriculumTopic(question) {
    closeCurriculumModal();
    sendQuickPrompt(question);
  }

  // Markdown Formatter
  function renderFormattedText(text) {
    if (!text) return "";
    let safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    safe = safe.replace(/```(?:\\w+)?\\r?\\n([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    safe = safe.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
    safe = safe.replace(/\\r?\\n/g, '<br>');
    return safe;
  }

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `message-row ${role === 'user' ? 'user' : 'bot'}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerText = role === 'user' ? 'YOU' : 'AI';

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = renderFormattedText(text);

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatBox.appendChild(row);

    scrollToBottom();
  }

  async function sendMessage() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    appendMessage("user", text);
    conversationHistory.push({ role: "user", content: text });

    const sendBtn = document.getElementById("send-btn");
    sendBtn.disabled = true;

    // Typing Indicator
    const typingRow = document.createElement("div");
    typingRow.className = "message-row bot";
    typingRow.id = "typing-indicator";
    typingRow.innerHTML = '<div class="avatar">AI</div><div class="bubble" style="color: #9ca3af; font-style: italic;">Thinking stepwise...</div>';
    chatBox.appendChild(typingRow);
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: conversationHistory })
      });

      const data = await response.json();
      const indicator = document.getElementById("typing-indicator");
      if (indicator) indicator.remove();

      const answer = data.answer || "No response received.";
      appendMessage("assistant", answer);
      conversationHistory.push({ role: "assistant", content: answer });
    } catch (err) {
      const indicator = document.getElementById("typing-indicator");
      if (indicator) indicator.remove();
      appendMessage("assistant", "[Error]: Could not connect to model server.");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  }

  function sendQuickPrompt(promptText) {
    document.getElementById("chat-input").value = promptText;
    sendMessage();
  }

  function clearChat() {
    conversationHistory = [];
    chatBox.innerHTML = `
      <div class="message-row bot">
        <div class="avatar">AI</div>
        <div class="bubble">
          <strong>Conversation memory reset.</strong><br>
          What new DBMS concept would you like to explore?
        </div>
      </div>
    `;
    scrollToBottom();
  }
</script>
</body>
</html>
"""


class ChatbotHTTPHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            status_data = {
                "device": DEVICE_INFO,
                "adapter": ADAPTER_PATH,
                "status": MODEL_STATE,
                "message": MODEL_MESSAGE,
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode("utf-8"))
            except Exception:
                data = {}
            messages = data.get("messages", [])

            t0 = time.time()
            answer = generate_response(messages)
            latency = round(time.time() - t0, 2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"answer": answer, "latency": latency}).encode("utf-8"))
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[HTTP] {self.address_string()} - {args[0]}\n")


def bind_server_with_fallback(initial_port: int = 7860, max_attempts: int = 5):
    """Attempts to bind to initial_port; if in use, tries next port cleanly."""
    port = initial_port
    for _ in range(max_attempts):
        try:
            server_address = ("", port)
            httpd = ThreadedHTTPServer(server_address, ChatbotHTTPHandler)
            return httpd, port
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, 'winerror', 0) == 10048:
                print(f"[Notice] Port {port} is busy. Trying {port + 1}...")
                port += 1
            else:
                raise e
    raise RuntimeError(f"Could not bind to any port between {initial_port} and {port - 1}.")


def run_server(port: int = 7860, adapter_path: str = ""):
    if not adapter_path:
        adapter_path = find_latest_adapter_dir("outputs")

    # 1. Start HTTP server first so the web UI opens immediately
    httpd, active_port = bind_server_with_fallback(port)
    print("\n" + "=" * 58)
    print(f"  DBMS Tutor Web Chatbot UI is LIVE!")
    print(f"  Open in your browser: http://localhost:{active_port}")
    print(f"  Device: {DEVICE_INFO}")
    print(f"  Press Ctrl+C in this terminal to stop.")
    print("=" * 58 + "\n")

    # 2. Launch model initialization in background worker thread
    import threading
    loader_thread = threading.Thread(
        target=init_model,
        args=(adapter_path,),
        daemon=True,
    )
    loader_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web Chatbot server...")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Web Chatbot for DBMS Tutor.")
    parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860).")
    parser.add_argument("--adapter_path", type=str, default="", help="Path to LoRA adapter.")
    args = parser.parse_args()

    run_server(port=args.port, adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
