import argparse
import torch
from config import DeepSeekConfig
from deepseek import DeepSeekModelForCausalLM
from bpe_tokenizer import Tokenizer
from datacollator import ChatMlSpecialTokens, format_input_text
import json
import os
import sys
import traceback

def load_model(checkpoint_path: str, device: str):
    with open("config.json", "r") as f:
        config_dict = json.load(f)
    config = DeepSeekConfig(**config_dict)
    model = DeepSeekModelForCausalLM(config).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"✅ Loaded model from {checkpoint_path}")

    model.eval()
    return model

@torch.no_grad()
def stream_generate(model, tokenizer, prompt: str, max_new_tokens=100, temperature=0.8):
    """
    Generate and stream tokens one at a time from the model
    """
    prompt_formatted = format_input_text(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt_formatted)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    kv_cache = None
    generated = input_tensor

    for _ in range(max_new_tokens):
        logits, _, kv_cache = model(generated, past_key_value=kv_cache)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        generated = next_token  # feed only the new token next round

        decoded = tokenizer.decode(next_token[0].tolist())
        print(decoded, end="", flush=True)

        if next_token[0].item() == tokenizer.eos_token_id:
            break
    print("\n" + "-" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-path", type=str, default="~/data/deepseek_sft/8_bit_optimizer_ckpt.pt")
    args = parser.parse_args()

    tokenizer = Tokenizer("cl100k_base")
    tokenizer.add_special_tokens([
        ChatMlSpecialTokens().bos_token,
        ChatMlSpecialTokens().eos_token
    ])

    model = load_model(args.checkpoint_path, args.device)

    print("🧠 DeepSeek Chat — Type your prompt and press Enter (Ctrl+C to exit)\n")

    while True:
        try:
            prompt = input("👤 You: ")
            print("🤖 Model: ", end="", flush=True)
            stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
