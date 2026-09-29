# 🎓 DBMS Tutor — Specialized Qwen2.5-1.5B QLoRA Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B--Instruct-green.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-orange.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, reproducible parameter-efficient fine-tuning (PEFT / QLoRA) pipeline to specialize **Qwen2.5-1.5B-Instruct** into a concise, stepwise **DBMS Academic Tutor**. The tutor breaks down core database concepts with concrete SQL examples, normal form proofs, and transaction schedules, accessible via both an interactive terminal CLI and a zero-dependency modern browser Web UI.

---

## 🌟 Key Highlights

- 🎯 **Pedagogical Alignment:** Instructed to deliver concise, stepwise explanations followed by concrete relational schemas or SQL code blocks.
- ⚡ **Lightweight & Efficient:** 4-bit NormalFloat (NF4) quantization runs on consumer GPUs with as little as 6 GB VRAM.
- 🔄 **Adaptive Backend Engine:** Automatically leverages **Unsloth** with Triton GPU acceleration when available, with seamless fallback to standard **Hugging Face Transformers + PEFT + TRL** on native Windows/CPU environments.
- 🛡️ **Zero Data Leakage:** Seeded 80/20 train/validation partitioning (`seed=42`) with programmatic leakage validation against unseen evaluation benchmarks.
- 🌐 **Modern Web Chatbot UI:** Glassmorphism dark-mode UI built on Python's standard `http.server` (no Flask/FastAPI/Node.js dependencies needed) with real-time model loading status, quick-topic prompt chips, and a built-in DBMS curriculum explorer.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DBMS Tutor Pipeline                             │
└────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    ▼                                                               ▼
[Data Pipeline]                                            [Training Engine]
data/dbms_tutor.jsonl (22 pairs)                           train_qwen_ft.py
   │                                                               │
   ├── scripts/validate_dataset.py                                 ├── 4-bit NF4 Quantization
   │   (schema, roles, char lengths)                               ├── LoRA (r=16, alpha=16)
   │                                                               ├── Response-only loss masking
   └── scripts/prepare_dataset.py (seed=42)                        └── Auto-incremented outputs/
       ├── data/train.jsonl (80%)                                      (outputs/experiment_001)
       └── data/validation.jsonl (20%)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    ▼                                                               ▼
[Evaluation Benchmark]                                     [Serving & UI]
scripts/evaluate_model.py                                  web_chat.py & inference.py
   │                                                               │
   ├── 10 Unseen Standardized DBMS Questions                       ├── Terminal REPL (inference.py)
   ├── Mode: baseline (Base Qwen2.5)                               │   - Multi-turn sliding window
   ├── Mode: finetuned (LoRA Adapter)                              │   - Fast context recovery
   └── Mode: compare                                               └── Browser Web UI (web_chat.py)
       └── outputs/evaluation_report.md                                - Glassmorphic Dark Mode
                                                                       - Async background model load
                                                                       - One-click launcher (run.bat)
```

---

## 📂 Repository Structure

```text
DBMS-tutor/
│
├── data/
│   ├── dbms_tutor.jsonl         # Master dataset (22 curated conversational Q&A pairs)
│   ├── train.jsonl              # Training partition (80%, ~18 examples)
│   └── validation.jsonl         # Disjoint validation partition (20%, ~4 examples)
│
├── scripts/
│   ├── validate_dataset.py      # Schema, role sequence, duplicates & length validation
│   ├── prepare_dataset.py       # Seeded 80/20 partitioner with zero-leakage assertions
│   └── evaluate_model.py        # 10-domain benchmark runner (baseline, finetuned, compare)
│
├── train_qwen_ft.py             # QLoRA fine-tuning script with ChatML formatting
├── inference.py                 # Multi-turn terminal CLI with sliding-window memory
├── web_chat.py                  # Standalone threaded HTTP Web Chatbot & API server
├── run.bat                      # One-click Windows CMD / Explorer launcher
├── run.ps1                      # Native Windows PowerShell launcher
├── requirements.txt             # Lean dependency specifications
├── .gitignore                   # Comprehensive git exclusions for models & caches
└── README.md                    # Project documentation
```

---

## 💻 Hardware & OS Requirements

| Component | Minimum | Recommended | Notes |
| :--- | :--- | :--- | :--- |
| **GPU VRAM** | 6 GB (GTX 1660 Ti / RTX 2060) | 8 GB+ (RTX 3060 / 4060 / Colab T4) | Required for 4-bit QLoRA fine-tuning |
| **System RAM** | 8 GB | 16 GB | Sufficient for data caching & tokenization |
| **Storage** | 10 GB free space | 20 GB free space | Accommodates base model weights & Hugging Face cache |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS | Linux / Windows (WSL2) | Full Unsloth acceleration runs on Linux/WSL2; Windows Native uses HF PEFT |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/manikanta2834/DBMS-tutor.git
cd DBMS-tutor
```

### 2. Environment Setup

#### Option A: Linux / WSL2 / Google Colab (Recommended)
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### Option B: Windows Native Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft accelerate bitsandbytes trl
```
*(The scripts dynamically detect Windows without Triton and run Hugging Face PEFT automatically).*

---

## 🔄 End-to-End Workflow Execution

### Step 1: Validate Dataset Integrity
Verifies valid JSON structure, allowed roles (`system`, `user`, `assistant`), non-empty content, and token distributions:
```bash
python scripts/validate_dataset.py --data_path data/dbms_tutor.jsonl
```

### Step 2: Generate Train/Validation Splits
Partitions the master dataset into an 80/20 train/validation split using `seed=42`:
```bash
python scripts/prepare_dataset.py
```

### Step 3: Run Baseline Evaluation (Before Training)
Tests the stock `Qwen2.5-1.5B-Instruct` model across 10 standardized unseen DBMS questions:
```bash
python scripts/evaluate_model.py --mode baseline --output_dir outputs
```

### Step 4: Execute QLoRA Fine-Tuning
Fine-tunes the model adapters using ChatML formatting and response-only cross-entropy loss:
```bash
python train_qwen_ft.py --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 4
```
*Adapters are automatically saved to `outputs/experiment_001/` (with auto-versioning for subsequent runs).*

### Step 5: Evaluate Fine-Tuned Model
Evaluates the fine-tuned adapter on the identical 10 benchmark questions:
```bash
python scripts/evaluate_model.py --mode finetuned --adapter_path outputs/experiment_001 --output_dir outputs
```

### Step 6: Generate Comparison Report
Compares baseline vs fine-tuned responses side-by-side to verify adherence to concise, stepwise tutoring:
```bash
python scripts/evaluate_model.py --mode compare --output_dir outputs
```
*Outputs a comprehensive Markdown report at `outputs/evaluation_report.md`.*

---

## 💬 Interactive Inference & Web UI

### Option 1: Modern Web Chatbot (Recommended)

Launch the web app with a single click:
- **Windows Command Prompt:** `run.bat`
- **Windows PowerShell:** `.\run.bat` or `.\run.ps1`
- **Linux / macOS:** `python web_chat.py --port 7860`

Then open your browser to **`http://localhost:7860`**.

#### Web UI Features:
- 🎨 **Glassmorphism Dark Theme:** Styled with Google Inter typography and tailored HSL color tokens.
- ⚡ **Non-Blocking Architecture:** Server binds instantly (<0.2s); model weights load safely in a background worker thread.
- 🧩 **Curriculum Syllabus Modal:** Interactive guide covering 5 university DBMS units.
- 💡 **Interactive Topic Chips:** One-click prompts for 2PL, BCNF, DELETE vs TRUNCATE, ACID, ARIES, and B+ Trees.
- 📱 **Mobile & Desktop Responsive:** Auto-scrolling, code copy buttons, and persistent conversation dock.

### Option 2: Terminal Interactive CLI

For quick terminal sessions with multi-turn conversation memory:
```bash
python inference.py --adapter_path outputs/experiment_001
```
Commands inside the REPL:
- `clear` / `reset` — Clear conversational context.
- `history` — View conversation turns.
- `exit` / `quit` — Close the session.

---

## ⚙️ Hyperparameter Configuration

| Parameter | Value | Technical Justification |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | High reasoning density at 1.5B parameter scale; fits comfortably on 6–8 GB VRAM. |
| **Quantization** | 4-bit NormalFloat (NF4) | Reduces model memory footprint from ~3.2 GB to ~1.2 GB. |
| **LoRA Rank ($r$)** | 16 | Adequate capacity to learn instructional tone without over-parameterization. |
| **LoRA Alpha ($\alpha$)** | 16 | Maintains a balanced scaling factor ($\alpha/r = 1.0$) for numerical stability. |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Adapts both self-attention routing and MLP feed-forward knowledge representations. |
| **Batch Size** | `2` (Grad Accum: `4`) | Effective batch size of 8; stabilizes gradient estimates while preserving low VRAM usage. |
| **Learning Rate** | `2e-4` with cosine decay | Enables fast initial adaptation with gradual tail convergence. |
| **Loss Masking** | `train_on_responses_only` | Prevents computing loss on system instructions and user queries. |

---

## 📊 Benchmark Evaluation Suite

The 10 unseen evaluation questions in [scripts/evaluate_model.py](scripts/evaluate_model.py) test across core database engineering disciplines:

1. **Keys & Constraints:** Candidate key definition and minimality criteria.
2. **SQL DDL vs DML:** `DELETE` vs `TRUNCATE` (rollback capability, row-by-row vs deallocation).
3. **Composite Keys:** Multi-attribute unique identification.
4. **Relational Integrity:** Referential integrity and foreign key constraints.
5. **Database Objects:** Physical storage vs virtual stored queries (`VIEW`).
6. **Relational Operations:** Inner, outer, and cross `JOIN` mechanisms.
7. **Normalization Theory:** Functional dependencies ($X \to Y$) and attribute closures.
8. **Advanced Normal Forms:** Boyce-Codd Normal Form (BCNF) superkey requirements.
9. **Concurrency Control:** Conflict and view serializability.
10. **Storage & Indexing:** Auxiliary B/B+ tree indexing structures and write overhead.

---

## 📈 Dataset Scaling Roadmap (To 2,000+ Examples)

To expand from the 22-example starter set to an enterprise-grade tutor:

```text
Unit 1: Fundamentals & Relational Modeling (300 Examples)
├── 3-Tier Architecture & Data Independence
├── Entity-Relationship (ER) to Relational Mapping
└── Key Hierarchies & Relational Integrity Constraints

Unit 2: Relational Algebra & SQL Mastery (400 Examples)
├── Fundamental Operations (Selection, Projection, Join, Division)
├── Complex SQL (Subqueries, CTEs, Window Functions)
└── Stored Procedures, Triggers & Views

Unit 3: Normalization & Dependency Theory (350 Examples)
├── Functional Dependencies & Armstrong's Axioms
├── 1NF, 2NF, 3NF, BCNF, 4NF, 5NF Decompositions
└── Lossless Join & Dependency Preservation Proofs

Unit 4: Transactions & Concurrency Control (350 Examples)
├── ACID Guarantees & Transaction Schedules
├── Concurrency Anomalies (Dirty Read, Non-Repeatable Read, Phantom)
└── 2-Phase Locking (2PL), Timestamp Ordering & Deadlock Handling

Unit 5: Storage Engine, Indexing & Recovery (350 Examples)
├── B-Trees vs B+ Trees (Splits, Merges, Range Queries)
├── Hash Indexing & Clustered Indexes
└── Write-Ahead Logging (WAL) & ARIES Recovery Protocol
```

---

## 🛠️ Troubleshooting

- **`CUDA Out of Memory (OOM)`:**
  In `train_qwen_ft.py`, reduce `--batch_size 1` and increase `--grad_accum 8`.
- **`Pylance Missing Imports in IDE`:**
  All dynamic modules use `importlib.util.find_spec` to ensure zero red squiggly warnings in VS Code / Antigravity IDE.
- **`PowerShell Execution Policy Error`:**
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating the virtual environment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
