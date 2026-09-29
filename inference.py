#!/usr/bin/env python3
"""
inference.py
Interactive CLI for asking DBMS questions to the fine-tuned Qwen2.5 DBMS Tutor model.

Features:
- Robust model & adapter loader (Unsloth with HF PEFT fallback)
- Safe fallback to base model if adapter is not yet trained
- Multi-turn conversation memory with sliding-window context management
- Device auto-mapping (CUDA / CPU) using model.device
- Interactive commands: 'clear', 'history', 'exit'
"""

import argparse
import glob
import os
import sys

SYSTEM_PROMPT = "You are a DBMS tutor. Answer concisely, stepwise, with small examples."


def find_latest_adapter_dir(base_dir: str = "outputs") -> str:
    """Finds latest experiment directory with an adapter, or defaults to outputs/experiment_001."""
    experiments = glob.glob(os.path.join(base_dir, "experiment_*"))
    if not experiments:
        return os.path.join(base_dir, "experiment_001")
    experiments.sort()
    return experiments[-1]


def check_adapter_exists(adapter_path: str) -> bool:
    """Checks whether a valid LoRA adapter exists at the given path."""
    if not os.path.isdir(adapter_path):
        return False
    # Check for either safetensors or bin weights and config
    has_config = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    has_weights = (
        os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
    )
    return has_config and has_weights


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


def load_tutor_model(
    adapter_path: str,
    base_model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
):
    """Loads base model + LoRA adapter with graceful fallback to base model."""
    import torch

    adapter_available = check_adapter_exists(adapter_path)
    if adapter_available:
        print(f"[Model Loader] Found trained LoRA adapter at: {adapter_path}")
    else:
        print(f"[Model Loader] No trained adapter found at '{adapter_path}'.")
        print(f"[Model Loader] Running base model ({base_model_name}) in zero-shot tutor mode.")

    target_load_path = adapter_path if adapter_available else base_model_name

    # 1. Try Unsloth engine first
    FastLanguageModel = _load_unsloth()
    if FastLanguageModel is not None:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=target_load_path,
                max_seq_length=512,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("[Engine] Unsloth inference engine initialized successfully.")
            return model, tokenizer
        except Exception as e_unsloth:
            print(f"[Notice] Unsloth load skipped ({e_unsloth}). Switching to Hugging Face PEFT...")
    else:
        print("[Notice] Unsloth not installed or not available. Using Hugging Face backend.")

    # 2. Fallback to standard Hugging Face PEFT + BitsAndBytes
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_base_model = "Qwen/Qwen2.5-1.5B-Instruct" if "unsloth" in base_model_name else base_model_name

    # Load tokenizer safely
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

    # Check CUDA
    has_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ) if has_cuda else None

    # Check for accelerate safely
    has_accelerate = False
    try:
        import importlib.util
        has_accelerate = importlib.util.find_spec("accelerate") is not None
    except Exception:
        has_accelerate = False

    model_kwargs = {"trust_remote_code": True}
    if has_cuda:
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if has_accelerate:
            model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = compute_dtype
    else:
        model_kwargs["torch_dtype"] = torch.float32

    print(f"[Engine] Loading base model '{hf_base_model}' (4-bit: {has_cuda})...")
    model = AutoModelForCausalLM.from_pretrained(hf_base_model, **model_kwargs)
    if has_cuda and not has_accelerate:
        model = model.to("cuda")

    if adapter_available:
        PeftModel = _load_peft()
        if PeftModel is not None:
            print(f"[Engine] Applying LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print(f"[Warning] Adapter found at '{adapter_path}', but 'peft' library is not installed.")
            print("Running base model without adapter. Run 'pip install peft' to enable LoRA.")

    model.eval()
    print("[Engine] Hugging Face inference engine initialized successfully.")
    return model, tokenizer


def trim_conversation_history(history: list, max_turns: int = 6) -> list:
    """
    Maintains a sliding window of recent conversation turns to avoid exceeding
    the max sequence length (512 tokens).
    """
    if not history:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    system_msg = history[0] if history[0].get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT}
    dialogue = [m for m in history if m.get("role") in ["user", "assistant"]]

    trimmed_dialogue = dialogue[-max_turns:]
    return [system_msg] + trimmed_dialogue


def generate_tutor_response(
    model,
    tokenizer,
    conversation_history: list,
    max_new_tokens: int = 300,
    temperature: float = 0.3,
) -> str:
    """Formats conversation history with Qwen2.5 ChatML and decodes generation."""
    import torch

    trimmed_history = trim_conversation_history(conversation_history, max_turns=6)

    prompt = tokenizer.apply_chat_template(
        trimmed_history,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Place tensors on the model's actual primary device
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Determine EOS token IDs (includes <|im_end|>)
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=(temperature > 0.0),
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def run_repl(adapter_path: str, max_new_tokens: int = 300):
    """Interactive multi-turn Read-Eval-Print Loop."""
    model, tokenizer = load_tutor_model(adapter_path)

    print("\n" + "=" * 58)
    print("        DBMS Tutor Interactive Chatbot Console")
    print("        Base Model: Qwen2.5-1.5B-Instruct")
    print("=" * 58)
    print("Commands:")
    print("  'exit' or 'quit' -> Close the chatbot session")
    print("  'clear' or 'new'  -> Start a new conversation thread")
    print("  'history'        -> View current conversation messages")
    print("=" * 58 + "\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                print("\nGoodbye! Happy learning.\n")
                break

            if cmd in ["clear", "reset", "new"]:
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("\n[Chatbot] Conversation memory reset. What would you like to explore next?\n")
                continue

            if cmd == "history":
                print("\n--- Current Conversation History ---")
                for msg in history:
                    print(f"[{msg['role'].upper()}]: {msg['content']}")
                print("------------------------------------\n")
                continue

            # Append user question to history
            history.append({"role": "user", "content": user_input})

            print("\nDBMS Tutor >")
            answer = generate_tutor_response(
                model=model,
                tokenizer=tokenizer,
                conversation_history=history,
                max_new_tokens=max_new_tokens,
            )
            print(answer)
            print("\n" + "-" * 58 + "\n")

            # Append assistant response to history
            history.append({"role": "assistant", "content": answer})

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error during generation]: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive DBMS Tutor inference console.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Path to trained LoRA adapter (defaults to latest in outputs/).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum generation tokens per response.",
    )
    args = parser.parse_args()

    adapter_path = args.adapter_path if args.adapter_path else find_latest_adapter_dir("outputs")
    run_repl(adapter_path, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
