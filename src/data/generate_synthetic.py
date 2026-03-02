"""
Generate synthetic Python instruction/response training data using Claude API.
Produces ChatML-format JSONL for fine-tuning Qwen2.5-Coder-0.5B with mlx-lm.

NOTE: For API-free generation, use generate_from_corpus.py (template-based)
or use Cursor's Claude to generate batches via Task subagents.
"""
import argparse
import json
import os
import random
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    raise

SYSTEM_PROMPT = (
    "You are a Python expert. Write clean, idiomatic, well-documented Python code. "
    "Follow PEP 8 conventions. Include type hints where appropriate. "
    "Explain your reasoning briefly before the code when helpful."
)

CATEGORIES = {
    "code_generation": {
        "weight": 0.30,
        "description": "Write a Python function/class from a specification",
        "prompts": [
            "Write a Python function that {task}.",
            "Implement a Python class that {task}.",
            "Create a Python function to {task}. Include type hints and a docstring.",
            "Write a Python generator that {task}.",
            "Implement {task} in Python using only the standard library.",
        ],
    },
    "debugging": {
        "weight": 0.18,
        "description": "Find and fix bugs in Python code",
        "prompts": [
            "The following Python code has a bug. Find and fix it:\n\n```python\n{code}\n```",
            "This Python function doesn't work correctly. Identify the issue and provide the corrected version:\n\n```python\n{code}\n```",
            "Debug this Python code and explain what was wrong:\n\n```python\n{code}\n```",
        ],
    },
    "code_explanation": {
        "weight": 0.18,
        "description": "Explain what Python code does",
        "prompts": [
            "Explain what this Python code does step by step:\n\n```python\n{code}\n```",
            "What does this Python function do? Explain its time and space complexity:\n\n```python\n{code}\n```",
            "Break down this Python code and explain each part:\n\n```python\n{code}\n```",
        ],
    },
    "code_review": {
        "weight": 0.12,
        "description": "Review Python code for improvements",
        "prompts": [
            "Review this Python code and suggest improvements:\n\n```python\n{code}\n```",
            "What are the potential issues with this Python code? How would you improve it?\n\n```python\n{code}\n```",
        ],
    },
    "refactoring": {
        "weight": 0.12,
        "description": "Refactor Python code for clarity/performance",
        "prompts": [
            "Refactor this Python code to be more readable and Pythonic:\n\n```python\n{code}\n```",
            "Optimize this Python code for better performance:\n\n```python\n{code}\n```",
        ],
    },
    "testing": {
        "weight": 0.10,
        "description": "Write unit tests for Python code",
        "prompts": [
            "Write comprehensive pytest tests for this function:\n\n```python\n{code}\n```",
            "Create unit tests for the following Python class using pytest:\n\n```python\n{code}\n```",
        ],
    },
}

CODE_GEN_TASKS = [
    "finds the longest common subsequence of two strings",
    "implements a binary search tree with insert, delete, and search operations",
    "validates an email address using regex",
    "reads a CSV file and computes column statistics (mean, median, std)",
    "implements a LRU cache with O(1) get and put operations",
    "merges two sorted linked lists into one sorted list",
    "finds all permutations of a string",
    "implements a simple HTTP server that serves static files",
    "parses command-line arguments with subcommands",
    "computes the edit distance between two strings using dynamic programming",
    "implements a thread-safe queue with timeout support",
    "finds the shortest path in a weighted graph using Dijkstra's algorithm",
    "implements a decorator that retries a function on failure with exponential backoff",
    "validates and parses a JSON schema",
    "implements a simple rate limiter using the token bucket algorithm",
    "flattens a nested dictionary into a single-level dictionary with dot-notation keys",
    "implements a trie data structure for autocomplete functionality",
    "creates a context manager for database transactions with rollback on failure",
    "computes the nth Fibonacci number using matrix exponentiation",
    "implements a simple pub/sub event system",
    "serializes and deserializes a binary tree",
    "implements a bloom filter for membership testing",
    "chunks an iterable into fixed-size groups with an optional fill value",
    "implements topological sort for a directed acyclic graph",
    "creates a simple state machine with transitions and guards",
    "implements a custom iterator that lazily evaluates a pipeline of transformations",
    "parses and evaluates simple mathematical expressions with operator precedence",
    "implements connection pooling for HTTP requests",
    "finds all cycles in a directed graph",
    "implements a simple template engine that replaces placeholders with values",
    "validates a credit card number using the Luhn algorithm",
    "implements memoization with a configurable cache size and TTL",
    "reads and writes Protocol Buffer-like messages without external dependencies",
    "implements a simple task scheduler with priority and dependencies",
    "computes the convex hull of a set of 2D points",
    "implements a ring buffer with fixed capacity",
    "finds the median of two sorted arrays in O(log(min(m,n))) time",
    "implements a concurrent web scraper with rate limiting",
    "creates a dataclass-like decorator that auto-generates __init__, __repr__, and __eq__",
    "implements consistent hashing for distributed systems",
    "builds a simple CLI calculator that supports variables and functions",
    "implements a prefix tree (trie) that supports wildcard matching",
    "creates a file watcher that triggers callbacks on file changes",
    "implements a simple in-memory key-value store with TTL expiration",
    "computes all strongly connected components of a directed graph using Tarjan's algorithm",
    "implements an async retry mechanism with configurable backoff strategies",
    "creates a simple dependency injection container",
    "implements a thread pool executor from scratch using threading primitives",
    "parses and generates cron expressions",
    "implements a simple diff algorithm for comparing two texts line by line",
]

BUGGY_CODE_EXAMPLES = [
    '''def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',

    '''def flatten(lst):
    result = []
    for item in lst:
        if type(item) == list:
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Bug: doesn't handle tuples, sets, or other iterables''',

    '''class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop(0)
    
    def peek(self):
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0''',

    '''def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# This will be extremely slow for large n''',

    '''def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

# Bug: doesn't handle nested dicts properly, just overwrites''',

    '''import threading

class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
    
    def get(self):
        return self.count

# Bug: not thread-safe''',

    '''def parse_date(date_str):
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    return year, month, day

# Bug: no validation, no error handling, assumes specific format''',

    '''def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

# Bug: doesn't work with unhashable types like dicts or lists''',
]

REVIEW_CODE_EXAMPLES = [
    '''def process_data(data):
    results = []
    for item in data:
        if item is not None:
            if isinstance(item, str):
                item = item.strip()
                if len(item) > 0:
                    results.append(item.lower())
            elif isinstance(item, (int, float)):
                if item > 0:
                    results.append(str(item))
    return results''',

    '''import os
import json

def load_config(path):
    f = open(path)
    data = json.load(f)
    return data''',

    '''def find_user(users, name):
    for i in range(len(users)):
        if users[i]["name"] == name:
            return users[i]
    return None''',

    '''class Database:
    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connection = None
    
    def connect(self):
        import psycopg2
        self.connection = psycopg2.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password
        )
    
    def query(self, sql):
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()''',
]


def generate_prompt(category: str) -> str:
    cat = CATEGORIES[category]
    template = random.choice(cat["prompts"])

    if category == "code_generation":
        task = random.choice(CODE_GEN_TASKS)
        return template.format(task=task)
    elif category in ("debugging",):
        code = random.choice(BUGGY_CODE_EXAMPLES)
        return template.format(code=code)
    elif category in ("code_explanation", "code_review", "refactoring", "testing"):
        code = random.choice(REVIEW_CODE_EXAMPLES + BUGGY_CODE_EXAMPLES)
        return template.format(code=code)
    return template


def pick_category() -> str:
    categories = list(CATEGORIES.keys())
    weights = [CATEGORIES[c]["weight"] for c in categories]
    return random.choices(categories, weights=weights, k=1)[0]


def call_claude(client, prompt: str, model: str = "claude-sonnet-4-20250514") -> str | None:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"API error: {e}")
        return None


def format_example(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Python training data via Claude API")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of examples to generate")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Max parallel API calls")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save progress every N examples")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it: export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "train.jsonl")

    existing = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                h = content_hash(ex["messages"][1]["content"])
                existing.add(h)
        print(f"Found {len(existing)} existing examples, continuing...")

    generated = len(existing)
    target = args.num_examples
    category_counts = {c: 0 for c in CATEGORIES}

    print(f"Generating {target - generated} examples using {args.model}...")
    print(f"Output: {output_path}")

    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(total=target, initial=generated, desc="Generating")

        while generated < target:
            batch_prompts = []
            for _ in range(min(args.batch_size, target - generated)):
                cat = pick_category()
                prompt = generate_prompt(cat)
                h = content_hash(prompt)
                if h not in existing:
                    batch_prompts.append((cat, prompt, h))
                    existing.add(h)

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {}
                for cat, prompt, h in batch_prompts:
                    future = executor.submit(call_claude, client, prompt, args.model)
                    futures[future] = (cat, prompt, h)

                for future in as_completed(futures):
                    cat, prompt, h = futures[future]
                    response = future.result()

                    if response and len(response) > 50:
                        example = format_example(prompt, response)
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        f.flush()
                        generated += 1
                        category_counts[cat] += 1
                        pbar.update(1)

        pbar.close()

    print(f"\nDone! Generated {generated} examples.")
    print("Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    split_train_val(args.output_dir, val_ratio=0.05)


def split_train_val(output_dir: str, val_ratio: float = 0.05):
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "valid.jsonl")

    with open(train_path, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)
    val_count = max(1, int(len(lines) * val_ratio))

    val_lines = lines[:val_count]
    train_lines = lines[val_count:]

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)

    print(f"Split: {len(train_lines)} train, {len(val_lines)} validation")


if __name__ == "__main__":
    main()
