"""
Evaluate a model on Python coding benchmarks.
Wraps mlx_lm evaluate and adds custom Python-specific tests.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CUSTOM_PYTHON_TESTS = [
    {
        "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
        "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765",
        "description": "Fibonacci sequence",
    },
    {
        "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome, ignoring case and non-alphanumeric characters.\"\"\"",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('hello') == False",
        "description": "Palindrome check",
    },
    {
        "prompt": "def flatten(lst: list) -> list:\n    \"\"\"Flatten a nested list of arbitrary depth.\"\"\"",
        "test": "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([]) == []\nassert flatten([[1], [[2]], [[[3]]]]) == [1, 2, 3]",
        "description": "Flatten nested list",
    },
    {
        "prompt": "def two_sum(nums: list[int], target: int) -> tuple[int, int]:\n    \"\"\"Return indices of two numbers that add up to target.\"\"\"",
        "test": "assert two_sum([2, 7, 11, 15], 9) == (0, 1)\nassert two_sum([3, 2, 4], 6) == (1, 2)",
        "description": "Two Sum",
    },
    {
        "prompt": "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"",
        "test": "assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted_lists([], [1, 2]) == [1, 2]\nassert merge_sorted_lists([1], []) == [1]",
        "description": "Merge sorted lists",
    },
]


def run_custom_tests(model_path: str, temperature: float = 0.0, max_tokens: int = 512):
    from mlx_lm import load, generate

    print(f"\n{'='*60}")
    print(f"Custom Python Coding Tests")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    passed = 0
    total = len(CUSTOM_PYTHON_TESTS)

    for i, test_case in enumerate(CUSTOM_PYTHON_TESTS):
        print(f"[{i+1}/{total}] {test_case['description']}...")

        completion = generate(
            model, tokenizer,
            prompt=test_case["prompt"],
            max_tokens=max_tokens,
            temp=temperature,
        )

        full_code = test_case["prompt"] + completion
        test_code = full_code + "\n\n" + test_case["test"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  PASS")
                passed += 1
            else:
                print(f"  FAIL: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  FAIL: timeout")
        except Exception as e:
            print(f"  FAIL: {e}")
        finally:
            os.unlink(tmp_path)

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"{'='*60}\n")
    return passed, total


def run_lm_eval(model_path: str, tasks: str = "humaneval"):
    print(f"\nRunning lm-eval on {tasks}...")
    cmd = [
        sys.executable, "-m", "mlx_lm", "evaluate",
        "--model", model_path,
        "--tasks", tasks,
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Python coding model")
    parser.add_argument("--model", type=str, required=True, help="Path to MLX-LM model")
    parser.add_argument("--custom-only", action="store_true", help="Only run custom tests")
    parser.add_argument("--lm-eval-tasks", type=str, default="humaneval",
                        help="lm-eval tasks to run")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_custom_tests(args.model, args.temperature)

    if not args.custom_only:
        try:
            run_lm_eval(args.model, args.lm_eval_tasks)
        except Exception as e:
            print(f"lm-eval not available: {e}")
            print("Install with: pip install lm-eval")


if __name__ == "__main__":
    main()
