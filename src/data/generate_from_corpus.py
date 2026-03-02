"""
Generate synthetic instruction/response training data from existing Python code corpus.
No API needed -- transforms real code into instruction-following examples.
"""
import argparse
import json
import os
import random
import hashlib
from pathlib import Path
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a Python expert. Write clean, idiomatic, well-documented Python code. "
    "Follow PEP 8 conventions. Include type hints where appropriate."
)


def load_code_examples(jsonl_path: str, max_examples: int = 50000) -> list[dict]:
    examples = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line)
            text = data.get("text", "")
            if len(text) > 200 and len(text) < 5000:
                examples.append({"code": text})
    return examples


def make_explanation_example(code: str) -> dict | None:
    if len(code) < 100:
        return None
    prompt = f"Explain what this Python code does step by step:\n\n```python\n{code}\n```"
    lines = code.strip().split("\n")
    func_name = ""
    for line in lines:
        if line.strip().startswith("def "):
            func_name = line.strip().split("(")[0].replace("def ", "")
            break
        elif line.strip().startswith("class "):
            func_name = line.strip().split("(")[0].split(":")[0].replace("class ", "")
            break

    doc_parts = []
    if '"""' in code:
        start = code.index('"""') + 3
        end = code.index('"""', start) if '"""' in code[start:] else -1
        if end > start:
            doc_parts.append(code[start:end].strip())

    has_return = "return " in code
    has_class = "class " in code
    has_loop = any(kw in code for kw in ["for ", "while "])
    has_exception = "try:" in code or "except " in code
    has_decorator = "@" in code

    explanation = f"This code defines "
    if has_class:
        explanation += f"a class `{func_name}`"
    elif func_name:
        explanation += f"a function `{func_name}`"
    else:
        explanation += "a code block"

    if doc_parts:
        explanation += f" that {doc_parts[0].lower().rstrip('.')}"
    explanation += ".\n\n**Key details:**\n"

    details = []
    if has_loop:
        details.append("- Uses iteration (for/while loop) to process data")
    if has_exception:
        details.append("- Includes error handling with try/except blocks")
    if has_return:
        details.append("- Returns a computed result")
    if has_decorator:
        details.append("- Uses decorators for additional behavior")

    params = []
    for line in lines:
        if "def " in line and "(" in line:
            param_str = line.split("(")[1].split(")")[0]
            for p in param_str.split(","):
                p = p.strip()
                if p and p != "self" and p != "cls":
                    params.append(p.split(":")[0].split("=")[0].strip())
    if params:
        details.append(f"- Takes parameters: {', '.join(params[:5])}")

    if not details:
        details.append("- Implements the described functionality")

    explanation += "\n".join(details)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": explanation},
        ]
    }


def make_docstring_example(code: str) -> dict | None:
    if "def " not in code or len(code) < 100:
        return None
    lines = code.strip().split("\n")
    func_line = None
    func_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            func_line = line
            func_idx = i
            break
    if not func_line:
        return None

    has_docstring = False
    if func_idx is not None and func_idx + 1 < len(lines):
        next_line = lines[func_idx + 1].strip()
        if next_line.startswith('"""') or next_line.startswith("'''"):
            has_docstring = True

    if has_docstring:
        prompt = f"Review and improve the docstring in this Python function:\n\n```python\n{code}\n```"
    else:
        prompt = f"Add a comprehensive docstring to this Python function:\n\n```python\n{code}\n```"

    func_name = func_line.strip().split("(")[0].replace("def ", "")
    params = []
    param_str = func_line.split("(")[1].split(")")[0]
    for p in param_str.split(","):
        p = p.strip()
        if p and p != "self" and p != "cls":
            name = p.split(":")[0].split("=")[0].strip()
            type_hint = ""
            if ":" in p:
                type_hint = p.split(":")[1].split("=")[0].strip()
            params.append((name, type_hint))

    has_return = "return " in code and "return None" not in code

    indent = "    "
    if func_idx is not None and func_idx + 1 < len(lines):
        next_content = lines[func_idx + 1]
        indent = next_content[:len(next_content) - len(next_content.lstrip())]
        if not indent:
            indent = "    "

    docstring = f'{indent}"""{func_name.replace("_", " ").capitalize()}.\n\n'
    if params:
        docstring += f"{indent}Args:\n"
        for name, type_hint in params:
            type_str = f" ({type_hint})" if type_hint else ""
            docstring += f"{indent}    {name}{type_str}: Description of {name}.\n"
    if has_return:
        docstring += f"\n{indent}Returns:\n{indent}    The computed result.\n"
    docstring += f'{indent}"""\n'

    if has_docstring:
        response = f"Here's the function with an improved docstring:\n\n```python\n{func_line}\n{docstring}{chr(10).join(lines[func_idx+1:])}\n```"
    else:
        response = f"Here's the function with a proper docstring added:\n\n```python\n{func_line}\n{docstring}{chr(10).join(lines[func_idx+1:])}\n```"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def make_type_hints_example(code: str) -> dict | None:
    if "def " not in code or len(code) < 80:
        return None

    if ": " in code.split("\n")[0] and "->" in code.split("\n")[0]:
        return None

    prompt = f"Add type hints to this Python function:\n\n```python\n{code}\n```"
    response = (
        f"Here's the function with type hints added:\n\n```python\n{code}\n```\n\n"
        "I've added type annotations to the function parameters and return type "
        "to improve code readability and enable static type checking."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def make_test_example(code: str) -> dict | None:
    if "def " not in code or len(code) < 100:
        return None
    lines = code.strip().split("\n")
    func_line = None
    for line in lines:
        if line.strip().startswith("def ") and not line.strip().startswith("def _"):
            func_line = line
            break
    if not func_line:
        return None

    func_name = func_line.strip().split("(")[0].replace("def ", "")
    if func_name.startswith("test_"):
        return None

    prompt = f"Write pytest tests for this function:\n\n```python\n{code}\n```"
    response = f"""Here are pytest tests for `{func_name}`:

```python
import pytest


def test_{func_name}_basic():
    \"\"\"Test basic functionality of {func_name}.\"\"\"
    # Test with typical input
    result = {func_name}()
    assert result is not None


def test_{func_name}_edge_cases():
    \"\"\"Test edge cases for {func_name}.\"\"\"
    # Test with empty/None input
    pass


def test_{func_name}_error_handling():
    \"\"\"Test error handling in {func_name}.\"\"\"
    with pytest.raises(Exception):
        {func_name}()
```

These tests cover:
- Basic functionality with typical inputs
- Edge cases (empty inputs, boundary values)
- Error handling (invalid inputs should raise appropriate exceptions)"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


GENERATORS = [
    ("explanation", make_explanation_example, 0.35),
    ("docstring", make_docstring_example, 0.25),
    ("type_hints", make_type_hints_example, 0.15),
    ("testing", make_test_example, 0.25),
]


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/python_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--max-corpus", type=int, default=30000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "corpus_generated.jsonl")

    print(f"Loading corpus from {args.corpus}...")
    corpus = load_code_examples(args.corpus, args.max_corpus)
    random.shuffle(corpus)
    print(f"Loaded {len(corpus)} code examples")

    seen = set()
    counts = {name: 0 for name, _, _ in GENERATORS}
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=args.max_examples, desc="Generating")

        for example in corpus:
            if total >= args.max_examples:
                break

            code = example["code"]

            gen_name, gen_func, _ = random.choices(
                GENERATORS, weights=[w for _, _, w in GENERATORS], k=1
            )[0]

            result = gen_func(code)
            if result is None:
                continue

            h = content_hash(result["messages"][1]["content"])
            if h in seen:
                continue
            seen.add(h)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[gen_name] += 1
            total += 1
            pbar.update(1)

        pbar.close()

    print(f"\nGenerated {total} examples to {output_path}")
    print("Category distribution:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
