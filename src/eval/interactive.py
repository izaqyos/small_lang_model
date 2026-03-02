"""
Interactive code generation with a trained model.
Supports both mlx-pretrain models and mlx-lm models.
"""
import argparse
import sys


def run_mlx_lm(model_path: str, temperature: float = 0.2):
    from mlx_lm import load, generate

    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("Model loaded. Type Python code prompts (Ctrl+D to exit).\n")

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=512,
                temp=temperature,
            )
            print(response)
            print()
        except EOFError:
            print("\nBye!")
            break
        except KeyboardInterrupt:
            print("\nBye!")
            break


def run_pretrain_model(run_name: str, temperature: float = 1.0):
    sys.path.insert(0, "mlx-pretrain")
    from train import Trainer

    print(f"Loading model from runs/{run_name}...")
    config_path = f"runs/{run_name}/config.yaml"
    trainer = Trainer(config_path, for_training=False)

    print("Model loaded. Type a code prompt (Ctrl+D to exit).\n")

    import mlx.core as mx

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue

            tokens = trainer.tokenizer.tokenize(prompt)
            tokens = [trainer.tokenizer.BOS_TOKEN] + tokens
            x = mx.array([tokens])

            generated = list(tokens)
            for _ in range(256):
                logits = trainer.model(x)
                logits = logits[:, -1, :] / temperature
                probs = mx.softmax(logits, axis=-1)
                next_token = mx.argmax(probs, axis=-1).item()

                if next_token == trainer.tokenizer.EOS_TOKEN:
                    break
                generated.append(next_token)
                x = mx.array([[next_token]])

            text = trainer.tokenizer.detokenize(generated[len(tokens):])
            print(text)
            print()
        except EOFError:
            print("\nBye!")
            break
        except KeyboardInterrupt:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive code generation")
    parser.add_argument("--model", type=str, help="Path to MLX-LM model directory")
    parser.add_argument("--run", type=str, help="Name of mlx-pretrain run")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if args.model:
        run_mlx_lm(args.model, args.temperature)
    elif args.run:
        run_pretrain_model(args.run, args.temperature)
    else:
        parser.error("Specify --model (MLX-LM path) or --run (mlx-pretrain run name)")


if __name__ == "__main__":
    main()
