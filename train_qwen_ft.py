#!/usr/bin/env python3
"""
train_qwen_ft.py
Fine-tunes unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit on conversational DBMS tutoring data.

Key features:
- CUDA & Environment detection (Windows, Linux, WSL, Colab)
- Automatic fallback from Unsloth to Hugging Face PEFT + TRL SFTTrainer
- LoRA configuration tailored to Qwen2.5 attention and MLP modules
- Response-only loss masking with Qwen2.5 ChatML template
- Automatic experiment directory versioning (outputs/experiment_001, 002, ...)
- Overfitting warnings and conservative regularization
"""

import argparse
import glob
import json
import os
import platform
import sys
from typing import Dict, List, Tuple


def detect_environment():
    """Detects and reports the execution environment and GPU availability."""
    os_name = platform.system()
    release = platform.release()

    is_wsl = "microsoft" in release.lower() or "wsl" in release.lower()
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    is_kaggle = os.path.exists("/kaggle")

    env_type = "Linux"
    if is_colab:
        env_type = "Google Colab"
    elif is_kaggle:
        env_type = "Kaggle"
    elif is_wsl:
        env_type = "Windows (WSL2)"
    elif os_name == "Windows":
        env_type = "Windows (Native)"
    elif os_name == "Darwin":
        env_type = "macOS"

    print("\n================ SYSTEM ENVIRONMENT ================")
    print(f"OS:          {os_name} ({release})")
    print(f"Environment: {env_type}")
    print(f"Python:      {platform.python_version()}")

    import torch
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        print(f"CUDA:        Available (Version {cuda_version})")
        print(f"GPU:         {gpu_name}")
        print(f"VRAM:        {vram_gb:.2f} GB")
        if vram_gb < 5.5:
            print("WARNING: GPU VRAM is under 6 GB. You may need to set batch_size=1.")
    else:
        print("CUDA:        NOT AVAILABLE")
        print("----------------------------------------------------")
        print("CRITICAL ERROR: 4-bit QLoRA fine-tuning strictly requires an NVIDIA GPU with CUDA.")
        print("Quantized 4-bit matrix multiplication kernels (bitsandbytes) do NOT run on CPU.")
        print("To run locally on Windows, ensure an NVIDIA GPU with CUDA drivers is installed,")
        print("or run within WSL2 (Ubuntu) or a free Google Colab T4 GPU instance.")
        print("=====================================================\n")
        sys.exit(1)

    print("=====================================================\n")
    return env_type, cuda_available


def get_next_experiment_dir(base_dir: str = "outputs") -> str:
    """Finds next available experiment directory (e.g. outputs/experiment_001)."""
    os.makedirs(base_dir, exist_ok=True)
    existing = glob.glob(os.path.join(base_dir, "experiment_*"))
    highest_idx = 0
    for path in existing:
        folder_name = os.path.basename(path)
        parts = folder_name.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            highest_idx = max(highest_idx, int(parts[1]))

    next_idx = highest_idx + 1
    exp_dir = os.path.join(base_dir, f"experiment_{next_idx:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def load_jsonl_dataset(file_path: str) -> List[dict]:
    """Loads JSONL conversation records."""
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset file does not exist: {file_path}")
        sys.exit(1)

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def train(
    train_path: str = "data/train.jsonl",
    val_path: str = "data/validation.jsonl",
    base_model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length: int = 512,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "",
):
    detect_environment()

    if not output_dir:
        output_dir = get_next_experiment_dir("outputs")
    print(f"Target experiment directory: {output_dir}\n")

    train_data = load_jsonl_dataset(train_path)
    val_data = load_jsonl_dataset(val_path)

    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples.")

    if len(train_data) < 100:
        print("\n" + "!" * 56)
        print(f"WARNING: Very small dataset ({len(train_data)} examples).")
        print("The model may memorize training examples.")
        print("Use unseen evaluation questions in scripts/evaluate_model.py.")
        print("!" * 56 + "\n")

    # Try Unsloth first
    use_unsloth = False
    try:
        import importlib
        unsloth_mod = importlib.import_module("unsloth")
        unsloth_chat = importlib.import_module("unsloth.chat_templates")
        FastLanguageModel = getattr(unsloth_mod, "FastLanguageModel")
        get_chat_template = getattr(unsloth_chat, "get_chat_template")
        train_on_responses_only = getattr(unsloth_chat, "train_on_responses_only")
        use_unsloth = True
        print("[Backend] Using Unsloth accelerated kernel backend.")
    except Exception as e:
        print(f"[Backend] Unsloth import note: {e}")
        print("[Backend] Proceeding with standard Hugging Face PEFT + TRL backend.")

    import torch
    import importlib

    try:
        datasets_mod = importlib.import_module("datasets")
        Dataset = getattr(datasets_mod, "Dataset")
    except ImportError:
        print("ERROR: Required package 'datasets' is not installed.")
        print("Please run: pip install datasets")
        sys.exit(1)

    try:
        trl_mod = importlib.import_module("trl")
        SFTTrainer = getattr(trl_mod, "SFTTrainer")
        SFTConfig = getattr(trl_mod, "SFTConfig")
    except ImportError:
        print("ERROR: Required package 'trl' is not installed.")
        print("Please run: pip install trl")
        sys.exit(1)

    if use_unsloth:
        # --- Unsloth Pipeline ---
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detection (float16 / bfloat16)
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-2.5",
        )

        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in convos
            ]
            return {"text": texts}

        train_ds = Dataset.from_list(train_data).map(formatting_prompts_func, batched=True)
        val_ds = Dataset.from_list(val_data).map(formatting_prompts_func, batched=True)

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=max_seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=training_args,
        )

        # Apply response-only loss masking
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    else:
        # --- Standard Hugging Face PEFT + TRL Fallback ---
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        try:
            import importlib
            peft_mod = importlib.import_module("peft")
            LoraConfig = getattr(peft_mod, "LoraConfig")
            get_peft_model = getattr(peft_mod, "get_peft_model")
            prepare_model_for_kbit_training = getattr(peft_mod, "prepare_model_for_kbit_training")
        except ImportError:
            print("ERROR: Hugging Face PEFT fallback requires the 'peft' package.")
            print("Please run: pip install peft")
            sys.exit(1)

        hf_base_model = "Qwen/Qwen2.5-1.5B-Instruct" if "unsloth" in base_model_name else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(hf_base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            hf_base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        def formatting_func(example):
            return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to="none",
            max_seq_length=max_seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            formatting_func=formatting_func,
            args=training_args,
        )

    print("\nStarting LoRA Fine-Tuning...")
    train_result = trainer.train()

    # Save LoRA adapter & Tokenizer
    print(f"\nSaving LoRA adapter and tokenizer to: {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    config_info = {
        "base_model": base_model_name,
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": 16,
        "lora_alpha": 16,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
    }

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    print("\n================ TRAINING FINISHED ================")
    print(f"LoRA Adapter saved to: {output_dir}")
    print(f"Train loss:            {metrics.get('train_loss', 'N/A')}")
    print("Next step: Run baseline vs fine-tuned evaluation:")
    print(f"  python scripts/evaluate_model.py --mode finetuned --adapter_path {output_dir}")
    print("  python scripts/evaluate_model.py --mode compare")
    print("===================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on DBMS Tutor dataset.")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        base_model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
