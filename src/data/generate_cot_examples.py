"""
Generate chain-of-thought (CoT) training examples for debugging and code reasoning.
Each example includes step-by-step execution tracing, not just surface-level descriptions.

These are designed to teach a model to actually trace through code logic rather than
pattern-match debugging responses.

Usage:
    python src/data/generate_cot_examples.py --output data/synthetic/cot_examples.jsonl
"""
import argparse
import json
import os
import random
import hashlib

SYSTEM_PROMPT = (
    "You are a Python expert. When debugging, always trace through the code with a "
    "concrete example before identifying the bug. Show your reasoning step by step. "
    "When writing code, explain your approach before implementation."
)

PROMPT_TEMPLATES_DEBUG = [
    "Find the bug in this code and explain the fix:\n\n```python\n{code}\n```",
    "This code has a subtle bug. Trace through it with an example to find the issue:\n\n```python\n{code}\n```",
    "Debug this Python code. Walk through the execution step by step:\n\n```python\n{code}\n```",
]

PROMPT_TEMPLATES_TRACE = [
    "Trace through this code with the input `{input_expr}` and show the output:\n\n```python\n{code}\n```",
    "What does this code return when called with `{input_expr}`? Show each step:\n\n```python\n{code}\n```",
]

PROMPT_TEMPLATES_REVIEW = [
    "Review this code for correctness and edge cases:\n\n```python\n{code}\n```",
    "What are the potential issues with this code? Analyze step by step:\n\n```python\n{code}\n```",
]

BUG_PATTERNS = [
    {
        "name": "merge_sorted_missing_tail",
        "code": """def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result""",
        "test_input": "merge_sorted([1, 3, 5], [2, 4])",
        "trace": """Let me trace through with `merge_sorted([1, 3, 5], [2, 4])`:

- i=0, j=0: a[0]=1 < b[0]=2 → append 1, i=1. result=[1]
- i=1, j=0: a[1]=3 > b[0]=2 → append 2, j=1. result=[1, 2]
- i=1, j=1: a[1]=3 < b[1]=4 → append 3, i=2. result=[1, 2, 3]
- i=2, j=1: a[2]=5 > b[1]=4 → append 4, j=2. result=[1, 2, 3, 4]
- j=2 == len(b)=2 → while loop exits

**Bug:** a[2]=5 was never appended! Returns [1, 2, 3, 4] instead of [1, 2, 3, 4, 5].

The while loop exits when *either* list is exhausted, but the remaining elements from the other list are silently dropped.""",
        "fix": """```python
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
```

Added `result.extend(a[i:])` and `result.extend(b[j:])` to append any remaining elements after the loop.""",
    },
    {
        "name": "binary_search_integer_division",
        "code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        "test_input": "binary_search([1, 3, 5, 7, 9], 5)",
        "trace": """Let me trace through with `binary_search([1, 3, 5, 7, 9], 5)`:

- left=0, right=4
- mid = (0 + 4) / 2 = 2.0 (float!)
- arr[2.0] → TypeError: list indices must be integers, not float

**Bug:** Using `/` (true division) instead of `//` (floor division). In Python 3, `/` always returns a float, even for integer operands. `4 / 2 = 2.0`, not `2`.

This causes a TypeError when used as a list index.""",
        "fix": """```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

Changed `/ 2` to `// 2` for integer floor division.""",
    },
    {
        "name": "binary_search_operator_precedence",
        "code": """def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + hi // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1""",
        "test_input": "binary_search([1, 3, 5, 7, 9], 3)",
        "trace": """Let me trace through with `binary_search([1, 3, 5, 7, 9], 3)`:

- lo=0, hi=5
- mid = 0 + 5 // 2 = 0 + 2 = 2 (should be (0+5)//2 = 2 — happens to be correct here)
- arr[2]=5 > 3 → hi=2

- lo=0, hi=2
- mid = 0 + 2 // 2 = 0 + 1 = 1 (should be (0+2)//2 = 1 — still correct)
- arr[1]=3 == 3 → return 1 ✓

Seems fine? Let's try `binary_search([1, 3, 5, 7, 9], 7)`:

- lo=0, hi=5
- mid = 0 + 5 // 2 = 2
- arr[2]=5 < 7 → lo=3

- lo=3, hi=5
- mid = 3 + 5 // 2 = 3 + 2 = 5 ← **WRONG!** Should be (3+5)//2 = 4
- arr[5] → IndexError! Array only has indices 0-4.

**Bug:** Operator precedence. `lo + hi // 2` is parsed as `lo + (hi // 2)`, not `(lo + hi) // 2`. The `//` operator has higher precedence than `+`.""",
        "fix": """```python
def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1
```

Added parentheses: `(lo + hi) // 2` to ensure addition happens before division.""",
    },
    {
        "name": "mutable_default_argument",
        "code": """def add_item(item, lst=[]):
    lst.append(item)
    return lst""",
        "test_input": "add_item(1); add_item(2); add_item(3)",
        "trace": """Let me trace three consecutive calls:

Call 1: `add_item(1)`
- lst defaults to [] (the default object)
- lst.append(1) → lst is now [1]
- Returns [1] ✓

Call 2: `add_item(2)`
- lst defaults to... the SAME list object from call 1, which is [1]
- lst.append(2) → lst is now [1, 2]
- Returns [1, 2] ← Expected [2]!

Call 3: `add_item(3)`
- lst is still the same object: [1, 2]
- lst.append(3) → [1, 2, 3]
- Returns [1, 2, 3] ← Expected [3]!

**Bug:** Mutable default argument. Python evaluates default arguments once at function definition time, not at each call. The same list object is reused across all calls that don't provide an explicit `lst`.""",
        "fix": """```python
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

Use `None` as the default and create a new list inside the function body. This ensures each call gets its own fresh list.""",
    },
    {
        "name": "late_binding_closure",
        "code": """def make_multipliers():
    multipliers = []
    for i in range(5):
        multipliers.append(lambda x: x * i)
    return multipliers

fns = make_multipliers()
print([f(2) for f in fns])""",
        "test_input": "make_multipliers()",
        "trace": """Let me trace through this:

The loop creates 5 lambda functions: `lambda x: x * i` for i in 0..4.

Expected output: [0, 2, 4, 6, 8] (each function multiplies by its index)

Actual execution:
- After the loop, `i = 4` (the loop variable's final value)
- Each lambda captures `i` by **reference**, not by value
- When we call `f(2)` for any function, it looks up `i` and finds 4
- So ALL functions compute `x * 4`
- Output: [8, 8, 8, 8, 8]

**Bug:** Late binding in closures. The lambda captures the variable `i` itself, not the value of `i` at the time the lambda was created. By the time the lambdas are called, the loop has finished and `i == 4` for all of them.""",
        "fix": """```python
def make_multipliers():
    multipliers = []
    for i in range(5):
        multipliers.append(lambda x, i=i: x * i)
    return multipliers
```

Use a default argument `i=i` to capture the current value of `i` at each iteration. Default arguments are evaluated at function definition time, so each lambda gets its own copy.""",
    },
    {
        "name": "shallow_copy_nested",
        "code": """def duplicate_grid(grid):
    return grid.copy()

original = [[1, 2], [3, 4]]
copy = duplicate_grid(original)
copy[0][0] = 99
print(original)""",
        "test_input": "duplicate_grid([[1, 2], [3, 4]])",
        "trace": """Let me trace through:

- original = [[1, 2], [3, 4]]
- copy = grid.copy() → creates a shallow copy

A shallow copy creates a new outer list, but the inner lists are the SAME objects:
- original[0] and copy[0] point to the same [1, 2] list
- original[1] and copy[1] point to the same [3, 4] list

- copy[0][0] = 99 → modifies the shared inner list
- print(original) → [[99, 2], [3, 4]] ← Original was modified!

**Bug:** `.copy()` only creates a shallow copy. For nested structures, inner objects are shared references. Modifying `copy[0][0]` also modifies `original[0][0]` because they point to the same list.""",
        "fix": """```python
import copy

def duplicate_grid(grid):
    return copy.deepcopy(grid)
```

Use `copy.deepcopy()` to recursively copy all nested objects. Alternatively, for a list of lists specifically: `[row[:] for row in grid]`.""",
    },
    {
        "name": "dict_iteration_modification",
        "code": """def remove_zeros(d):
    for key in d:
        if d[key] == 0:
            del d[key]
    return d""",
        "test_input": "remove_zeros({'a': 1, 'b': 0, 'c': 3, 'd': 0})",
        "trace": """Let me trace with `{'a': 1, 'b': 0, 'c': 3, 'd': 0}`:

- Iterating: key='a', d['a']=1, not zero, skip
- key='b', d['b']=0, delete 'b' → RuntimeError!

Python raises `RuntimeError: dictionary changed size during iteration`.

You cannot add or remove keys from a dictionary while iterating over it. The iterator becomes invalid when the dictionary structure changes.

**Bug:** Modifying a dictionary's size during iteration. The `for key in d` creates an iterator over the dictionary, and `del d[key]` invalidates it.""",
        "fix": """```python
def remove_zeros(d):
    keys_to_remove = [key for key in d if d[key] == 0]
    for key in keys_to_remove:
        del d[key]
    return d
```

Collect the keys to remove first, then delete them in a separate loop. Alternatively: `return {k: v for k, v in d.items() if v != 0}`.""",
    },
    {
        "name": "off_by_one_range",
        "code": """def is_sorted(arr):
    for i in range(len(arr)):
        if arr[i] > arr[i + 1]:
            return False
    return True""",
        "test_input": "is_sorted([1, 2, 3, 4, 5])",
        "trace": """Let me trace with `is_sorted([1, 2, 3, 4, 5])`:

- i=0: arr[0]=1 > arr[1]=2? No, continue
- i=1: arr[1]=2 > arr[2]=3? No, continue
- i=2: arr[2]=3 > arr[3]=4? No, continue
- i=3: arr[3]=4 > arr[4]=5? No, continue
- i=4: arr[4]=5 > arr[5]=? → IndexError! Array has indices 0-4.

**Bug:** Off-by-one in `range(len(arr))`. Since we access `arr[i + 1]`, the loop should stop at `len(arr) - 1` to avoid accessing beyond the last element.""",
        "fix": """```python
def is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True
```

Changed `range(len(arr))` to `range(len(arr) - 1)`. The last valid comparison is `arr[-2]` vs `arr[-1]`.""",
    },
    {
        "name": "stack_wrong_end",
        "code": """class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[0]""",
        "test_input": "s = Stack(); s.push(1); s.push(2); s.push(3); s.pop()",
        "trace": """Let me trace:

- push(1): items = [1]
- push(2): items = [1, 2]
- push(3): items = [1, 2, 3]
- pop(): items.pop(0) → returns 1, items = [2, 3]

But a stack should return the LAST element pushed (LIFO). pop() should return 3, not 1.

- peek(): items[0] → returns 2 (the front), should return 3 (the top)

**Bug:** Using index 0 (front of list) instead of -1 (back of list). `pop(0)` and `items[0]` implement a queue (FIFO), not a stack (LIFO). `append()` adds to the end, so the "top" of the stack is the end of the list.""",
        "fix": """```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()  # pop from end (default)

    def peek(self):
        return self.items[-1]  # look at end
```

Changed `pop(0)` to `pop()` (defaults to last element) and `items[0]` to `items[-1]`. Also, `pop(0)` is O(n) because it shifts all elements, while `pop()` is O(1).""",
    },
    {
        "name": "floating_point_comparison",
        "code": """def calculate_total(prices):
    total = 0.0
    for price in prices:
        total += price
    return total

prices = [0.1, 0.2, 0.3]
if calculate_total(prices) == 0.6:
    print("Correct total")
else:
    print("Something is wrong")""",
        "test_input": "calculate_total([0.1, 0.2, 0.3])",
        "trace": """Let me trace:

- total = 0.0
- total += 0.1 → total = 0.1
- total += 0.2 → total = 0.30000000000000004 (not 0.3!)
- total += 0.3 → total = 0.6000000000000001 (not 0.6!)

So `calculate_total([0.1, 0.2, 0.3]) == 0.6` → `0.6000000000000001 == 0.6` → False

Output: "Something is wrong"

**Bug:** Floating point representation error. Most decimal fractions (like 0.1) cannot be represented exactly in binary floating point. Small rounding errors accumulate during addition. Direct equality comparison with `==` is unreliable for floats.""",
        "fix": """```python
import math

prices = [0.1, 0.2, 0.3]
if math.isclose(calculate_total(prices), 0.6):
    print("Correct total")
else:
    print("Something is wrong")
```

Use `math.isclose()` for float comparison (default tolerance: 1e-9). Alternatively, use `decimal.Decimal` for exact decimal arithmetic, or compare with `abs(a - b) < epsilon`.""",
    },
    {
        "name": "is_vs_equals",
        "code": """def check_value(x):
    if x is 1000:
        return "found"
    return "not found"

result = check_value(10 * 100)""",
        "test_input": "check_value(10 * 100)",
        "trace": """Let me trace:

- `10 * 100` evaluates to `1000`
- Inside the function: `x is 1000`
- `is` checks object identity (are they the same object in memory?)
- `==` checks value equality (do they have the same value?)
- Python caches small integers (-5 to 256), so `is` works for those
- But 1000 > 256, so `10 * 100` and the literal `1000` are different objects
- `x is 1000` → False
- Returns "not found" ← Wrong! 1000 equals 1000.

**Bug:** Using `is` for value comparison instead of `==`. `is` checks identity (same object), not equality (same value). It works for small integers by coincidence (Python caches -5 to 256) but fails for larger values.""",
        "fix": """```python
def check_value(x):
    if x == 1000:
        return "found"
    return "not found"
```

Use `==` for value comparison. Reserve `is` for singleton checks like `x is None`, `x is True`, `x is False`.""",
    },
    {
        "name": "generator_exhaustion",
        "code": """def get_evens(numbers):
    return (x for x in numbers if x % 2 == 0)

evens = get_evens([1, 2, 3, 4, 5, 6])
print(f"Count: {sum(1 for _ in evens)}")
print(f"Sum: {sum(evens)}")""",
        "test_input": "get_evens([1, 2, 3, 4, 5, 6])",
        "trace": """Let me trace:

- `get_evens` returns a generator (not a list) — note the parentheses `()` not `[]`
- evens = generator yielding 2, 4, 6

First print:
- `sum(1 for _ in evens)` iterates through the generator: 2, 4, 6
- Count = 3 ✓
- The generator is now exhausted (all items consumed)

Second print:
- `sum(evens)` tries to iterate through the generator again
- But the generator is already exhausted — it yields nothing
- Sum = 0 ← Wrong! Expected 12.

**Bug:** Generator exhaustion. A generator can only be iterated once. After the first `sum()` call consumes all elements, the generator is empty. The second `sum()` gets no elements.""",
        "fix": """```python
def get_evens(numbers):
    return [x for x in numbers if x % 2 == 0]  # list, not generator
```

Use a list comprehension `[...]` instead of a generator expression `(...)` if you need to iterate multiple times. Alternatively, convert to list: `evens = list(get_evens(...))`.""",
    },
    {
        "name": "recursive_default_accumulator",
        "code": """def flatten(nested, result=[]):
    for item in nested:
        if isinstance(item, list):
            flatten(item, result)
        else:
            result.append(item)
    return result""",
        "test_input": "flatten([1, [2, 3]]); flatten([4, [5]])",
        "trace": """Let me trace two calls:

Call 1: `flatten([1, [2, 3]])`
- result starts as [] (the default mutable object)
- item=1, not a list → result.append(1), result=[1]
- item=[2,3], is a list → flatten([2,3], result)
  - item=2 → result.append(2), result=[1,2]
  - item=3 → result.append(3), result=[1,2,3]
- Returns [1, 2, 3] ✓

Call 2: `flatten([4, [5]])`
- result defaults to... the SAME list from call 1: [1, 2, 3]
- item=4 → result.append(4), result=[1,2,3,4]
- item=[5] → flatten([5], result)
  - item=5 → result.append(5), result=[1,2,3,4,5]
- Returns [1, 2, 3, 4, 5] ← Expected [4, 5]!

**Bug:** Mutable default argument combined with recursion. The default `result=[]` is created once and reused across all top-level calls. Previous results accumulate.""",
        "fix": """```python
def flatten(nested, result=None):
    if result is None:
        result = []
    for item in nested:
        if isinstance(item, list):
            flatten(item, result)
        else:
            result.append(item)
    return result
```

Use `None` as default and create a new list inside the function.""",
    },
    {
        "name": "scope_variable_leaking",
        "code": """def process_items(items):
    results = []
    for item in items:
        try:
            value = int(item)
        except ValueError:
            pass
        results.append(value)
    return results""",
        "test_input": 'process_items(["1", "abc", "3"])',
        "trace": """Let me trace with `["1", "abc", "3"]`:

- item="1": int("1") succeeds, value=1, append 1. results=[1]
- item="abc": int("abc") raises ValueError, `pass` (value is NOT set to anything new)
  - value is still 1 from the previous iteration!
  - append 1 (the old value). results=[1, 1]
- item="3": int("3") succeeds, value=3, append 3. results=[1, 1, 3]

Returns [1, 1, 3] ← Expected either [1, 3] (skip invalid) or an error for "abc".

**Bug:** When the except clause runs `pass`, `value` retains its previous value from the last successful iteration. The `append` runs unconditionally, so it appends stale data for invalid items.""",
        "fix": """```python
def process_items(items):
    results = []
    for item in items:
        try:
            value = int(item)
        except ValueError:
            continue  # skip invalid items
        results.append(value)
    return results
```

Use `continue` instead of `pass` to skip the append for invalid items. Or move `results.append(value)` inside the try block.""",
    },
    {
        "name": "string_replace_not_inplace",
        "code": """def clean_text(text):
    text.strip()
    text.replace("  ", " ")
    text.lower()
    return text""",
        "test_input": 'clean_text("  Hello  World  ")',
        "trace": """Let me trace with `"  Hello  World  "`:

- text.strip() → returns "Hello  World" but text is NOT modified
- text.replace("  ", " ") → returns "Hello World" but text is NOT modified
- text.lower() → returns "  hello  world  " but text is NOT modified
- return text → returns the original "  Hello  World  "

**Bug:** Strings in Python are immutable. Methods like `strip()`, `replace()`, and `lower()` return NEW strings — they do not modify the original. The return values are being discarded.""",
        "fix": """```python
def clean_text(text):
    text = text.strip()
    text = text.replace("  ", " ")
    text = text.lower()
    return text
```

Assign the return values back to `text`. Or chain: `return text.strip().replace("  ", " ").lower()`.""",
    },
    {
        "name": "list_multiply_shared_reference",
        "code": """def create_grid(rows, cols, default=0):
    return [[default] * cols] * rows

grid = create_grid(3, 3)
grid[0][0] = 1
print(grid)""",
        "test_input": "create_grid(3, 3)",
        "trace": """Let me trace:

- `[0] * 3` → [0, 0, 0] (a single row)
- `[[0, 0, 0]] * 3` → [[0,0,0], [0,0,0], [0,0,0]]
  BUT all three rows are the SAME list object!

- grid[0][0] = 1 → modifies the shared list
- print(grid) → [[1, 0, 0], [1, 0, 0], [1, 0, 0]]

Expected [[1, 0, 0], [0, 0, 0], [0, 0, 0]] but ALL rows changed!

**Bug:** `[row] * n` creates n references to the SAME row object, not n independent copies. This is Python's list multiplication behavior — it copies references, not values. The inner `[default] * cols` is fine because integers are immutable, but the outer `* rows` replicates the same mutable list.""",
        "fix": """```python
def create_grid(rows, cols, default=0):
    return [[default] * cols for _ in range(rows)]
```

Use a list comprehension to create independent row lists. Each iteration of the comprehension creates a new `[default] * cols` list.""",
    },
    {
        "name": "max_subarray_reset_logic",
        "code": """def max_subarray_sum(arr):
    max_sum = 0
    current_sum = 0
    for num in arr:
        current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum
        if current_sum < 0:
            current_sum = 0
    return max_sum""",
        "test_input": "max_subarray_sum([-2, -3, -1, -5])",
        "trace": """Let me trace with `[-2, -3, -1, -5]` (all negative):

- max_sum=0, current_sum=0
- num=-2: current_sum=-2, -2 > 0? No. -2 < 0? Yes, reset current_sum=0
- num=-3: current_sum=-3, -3 > 0? No. -3 < 0? Yes, reset current_sum=0
- num=-1: current_sum=-1, -1 > 0? No. -1 < 0? Yes, reset current_sum=0
- num=-5: current_sum=-5, -5 > 0? No. -5 < 0? Yes, reset current_sum=0
- Returns 0

**Bug:** Returns 0 for all-negative arrays. The maximum subarray sum of [-2, -3, -1, -5] should be -1 (the subarray [-1]). Initializing `max_sum = 0` means negative sums are never recorded.""",
        "fix": """```python
def max_subarray_sum(arr):
    max_sum = float('-inf')
    current_sum = 0
    for num in arr:
        current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum
        if current_sum < 0:
            current_sum = 0
    return max_sum
```

Initialize `max_sum = float('-inf')` so any real sum (including negative) will be recorded as the maximum.""",
    },
    {
        "name": "fibonacci_exponential",
        "code": """def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)""",
        "test_input": "fibonacci(35)",
        "trace": """Let me trace the call tree for fibonacci(5):

```
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2) → fib(1) + fib(0) = 1
│   │   └── fib(1) = 1
│   └── fib(2) → fib(1) + fib(0) = 1
└── fib(3)
    ├── fib(2) → fib(1) + fib(0) = 1
    └── fib(1) = 1
```

fib(3) is computed twice, fib(2) is computed three times. For fib(35), this explodes to ~29 million calls. Time complexity is O(2^n).

**Bug:** Not a correctness bug — the function returns the right answer, but is impractically slow for n > 30. Each call branches into two recursive calls, creating an exponential explosion of redundant computation.""",
        "fix": """```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]
```

Add memoization to cache computed results. Now each fib(k) is computed at most once, giving O(n) time. Alternatively, use `@functools.lru_cache` or an iterative approach.""",
    },
    {
        "name": "two_sum_wrong_index",
        "code": """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], seen[complement]]
        seen[num] = i
    return []""",
        "test_input": "two_sum([2, 7, 11, 15], 9)",
        "trace": """Let me trace with `two_sum([2, 7, 11, 15], 9)`:

- i=0, num=2: complement=9-2=7, 7 not in seen. seen={2: 0}
- i=1, num=7: complement=9-7=2, 2 in seen! seen[2]=0
  - return [seen[2], seen[2]] → [0, 0]

Expected [0, 1] (indices of 2 and 7), but got [0, 0].

**Bug:** The return statement uses `seen[complement]` twice instead of `[seen[complement], i]`. It returns the index of the complement both times, missing the current index `i`.""",
        "fix": """```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

Changed the return to `[seen[complement], i]` — the stored index of the complement and the current index.""",
    },
    {
        "name": "bfs_visited_too_late",
        "code": """from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = set()
    result = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in graph.get(node, []):
            queue.append(neighbor)
    return result""",
        "test_input": "bfs({'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}, 'A')",
        "trace": """This code is functionally correct but has a performance problem. Let me trace with a fully connected graph A-B-C:

- queue=[A], visited={}
- Pop A, not visited → visited={A}, result=[A], enqueue B, C. queue=[B, C]
- Pop B, not visited → visited={A,B}, result=[A,B], enqueue A, C. queue=[C, A, C]
- Pop C, not visited → visited={A,B,C}, result=[A,B,C], enqueue A, B. queue=[A, C, A, B]
- Pop A, already visited → skip. queue=[C, A, B]
- Pop C, already visited → skip. queue=[A, B]
- Pop A, already visited → skip. queue=[B]
- Pop B, already visited → skip. queue=[]

Result is correct: [A, B, C]. But the queue grew to 5 elements when it only needed 3.

**Bug:** The visited check happens after dequeuing, but nodes are enqueued without checking if they're already visited or already in the queue. In dense graphs, this causes the queue to grow much larger than necessary — O(E) instead of O(V).""",
        "fix": """```python
from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = {start}
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result
```

Mark nodes as visited BEFORE enqueuing them, not after dequeuing. This prevents duplicates from entering the queue.""",
    },
    {
        "name": "reverse_linked_list_lost_pointer",
        "code": """class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        curr.next = prev
        prev = curr
        curr = curr.next
    return prev""",
        "test_input": "1 -> 2 -> 3 -> None",
        "trace": """Let me trace with list 1 → 2 → 3 → None:

Iteration 1: prev=None, curr=Node(1)
- curr.next = prev → Node(1).next = None (was pointing to Node(2))
- prev = curr → prev = Node(1)
- curr = curr.next → curr = None ← WRONG!

We just set curr.next to None in the previous line, so curr.next is None, not Node(2). We lost the reference to the rest of the list!

**Bug:** We overwrite `curr.next` before saving the reference to the next node. After `curr.next = prev`, the original next pointer is lost and `curr = curr.next` gets None.""",
        "fix": """```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next  # save next BEFORE overwriting
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

Save `curr.next` in a temporary variable before overwriting it. The correct order is: save next, reverse pointer, advance both pointers.""",
    },
    {
        "name": "exception_bare_except",
        "code": """import sys

def safe_divide(a, b):
    try:
        return a / b
    except:
        return 0""",
        "test_input": "safe_divide(10, 0); KeyboardInterrupt during safe_divide(10, 2)",
        "trace": """Obvious case: `safe_divide(10, 0)` catches ZeroDivisionError, returns 0. Seems fine.

But what about:
- User presses Ctrl+C during a long computation → `except:` catches KeyboardInterrupt
- System runs out of memory → `except:` catches MemoryError
- `safe_divide("10", "2")` → `except:` silently hides the TypeError
- Program is exiting → `except:` catches SystemExit

**Bug:** Bare `except:` catches ALL exceptions, including:
- KeyboardInterrupt (Ctrl+C — user wants to stop the program)
- SystemExit (sys.exit() — program should terminate)
- MemoryError (system is out of memory)

This makes the program impossible to interrupt and hides real errors.""",
        "fix": """```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
```

Catch only the specific exception you expect. Never use bare `except:` or `except Exception:` unless you have a very good reason and re-raise after logging.""",
    },
    {
        "name": "thread_unsafe_counter",
        "code": """import threading

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

counter = Counter()
threads = [threading.Thread(target=counter.increment) for _ in range(1000)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(counter.count)""",
        "test_input": "1000 threads each calling increment()",
        "trace": """Expected output: 1000. Actual output: varies (e.g., 987, 993, 1000, 991).

The issue is in `self.count += 1`, which is NOT atomic. It compiles to:
1. Read self.count (e.g., 42)
2. Add 1 (43)
3. Write back to self.count (43)

Race condition between Thread A and Thread B:
- Thread A reads count = 42
- Thread B reads count = 42 (before A writes back)
- Thread A writes count = 43
- Thread B writes count = 43 (overwrites A's update!)
- Two increments, but count only went up by 1.

**Bug:** `self.count += 1` is a read-modify-write sequence that is not thread-safe. Multiple threads can read the same value before any of them writes back, causing lost updates.""",
        "fix": """```python
import threading

class Counter:
    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.count += 1
```

Use a `threading.Lock()` to ensure only one thread executes the read-modify-write at a time. The `with` statement automatically acquires and releases the lock.""",
    },
    {
        "name": "sql_injection",
        "code": """import sqlite3

def get_user(db, username):
    cursor = db.cursor()
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()""",
        "test_input": "get_user(db, \"admin' OR '1'='1\")",
        "trace": """Let me trace with a malicious input: `username = "admin' OR '1'='1"`

The f-string builds:
```sql
SELECT * FROM users WHERE name = 'admin' OR '1'='1'
```

This query returns ALL users because `'1'='1'` is always true. The attacker can:
- `"'; DROP TABLE users; --"` → deletes the entire table
- `"' UNION SELECT password FROM users --"` → extracts passwords

**Bug:** SQL injection via string formatting. User input is interpolated directly into the SQL query, allowing the attacker to inject arbitrary SQL commands by closing the quote and adding their own clauses.""",
        "fix": """```python
def get_user(db, username):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE name = ?", (username,))
    return cursor.fetchone()
```

Use parameterized queries with `?` placeholders. The database driver handles escaping and quoting, making injection impossible.""",
    },
    {
        "name": "power_negative_exponent",
        "code": """def power(base, exp):
    if exp == 0:
        return 1
    result = base
    for _ in range(exp - 1):
        result *= base
    return result""",
        "test_input": "power(2, -3)",
        "trace": """Let me trace with `power(2, -3)`:

- exp=-3, not 0
- result = 2
- range(-3 - 1) = range(-4) → empty range (no iterations)
- Returns 2

Expected: 2^(-3) = 1/8 = 0.125, but got 2.

Also, `power(2, 0)` works (returns 1), and `power(2, 3)` works (returns 8).

**Bug:** The function doesn't handle negative exponents. `range(negative_number)` produces an empty sequence, so the loop never executes and the function returns `base` unchanged.""",
        "fix": """```python
def power(base, exp):
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    result = base
    for _ in range(exp - 1):
        result *= base
    return result
```

Added a check for negative exponents: compute the positive power and take the reciprocal. For even better performance, use exponentiation by squaring.""",
    },
    {
        "name": "walrus_scope_confusion",
        "code": """def find_first_long_word(words, min_length=5):
    result = None
    for word in words:
        if len(word) >= min_length:
            result = word
    return result""",
        "test_input": 'find_first_long_word(["hi", "hello", "world", "programming"])',
        "trace": """Let me trace with `["hi", "hello", "world", "programming"]`:

- word="hi": len("hi")=2, 2 >= 5? No
- word="hello": len("hello")=5, 5 >= 5? Yes → result = "hello"
- word="world": len("world")=5, 5 >= 5? Yes → result = "world"
- word="programming": len("programming")=11, 11 >= 5? Yes → result = "programming"

Returns "programming" ← Expected "hello" (the FIRST long word).

**Bug:** The loop doesn't stop after finding the first match. It continues iterating and overwrites `result` with each subsequent match, ultimately returning the LAST long word.""",
        "fix": """```python
def find_first_long_word(words, min_length=5):
    for word in words:
        if len(word) >= min_length:
            return word
    return None
```

Return immediately when the first match is found. No need for a `result` variable — early return is cleaner and more efficient.""",
    },
    {
        "name": "deepcopy_vs_assignment",
        "code": """def remove_negatives(matrix):
    clean = matrix
    for i in range(len(clean)):
        for j in range(len(clean[i])):
            if clean[i][j] < 0:
                clean[i][j] = 0
    return clean

data = [[1, -2, 3], [-4, 5, -6]]
cleaned = remove_negatives(data)
print(data)""",
        "test_input": "remove_negatives([[1, -2, 3], [-4, 5, -6]])",
        "trace": """Let me trace:

- `clean = matrix` → clean and matrix point to the SAME object (no copy!)
- Loop replaces negatives with 0 in `clean`
- But `clean` IS `matrix`, so the original is also modified

print(data) → [[1, 0, 3], [0, 5, 0]] ← Original was mutated!
Expected data to still be [[1, -2, 3], [-4, 5, -6]]

**Bug:** `clean = matrix` does not create a copy. It creates another reference to the same list. All modifications to `clean` directly affect `matrix` (and the caller's `data`).""",
        "fix": """```python
import copy

def remove_negatives(matrix):
    clean = copy.deepcopy(matrix)
    for i in range(len(clean)):
        for j in range(len(clean[i])):
            if clean[i][j] < 0:
                clean[i][j] = 0
    return clean
```

Use `copy.deepcopy()` for nested structures. Or return a new list: `return [[max(0, x) for x in row] for row in matrix]`.""",
    },
    {
        "name": "enumerate_wrong_unpack",
        "code": """def find_duplicates(lst):
    seen = set()
    duplicates = []
    for item, index in enumerate(lst):
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return duplicates""",
        "test_input": 'find_duplicates(["a", "b", "a", "c"])',
        "trace": """Let me trace with `["a", "b", "a", "c"]`:

`enumerate(lst)` yields (index, item) tuples: (0, "a"), (1, "b"), (2, "a"), (3, "c")

But the unpacking is `for item, index in enumerate(lst)`:
- First iteration: item=0, index="a" ← Swapped!
- Checking: 0 in seen? No. seen={0}
- Second iteration: item=1, index="b". 1 in seen? No. seen={0, 1}
- Third iteration: item=2, index="a". 2 in seen? No. seen={0, 1, 2}
- Fourth iteration: item=3, index="c". 3 in seen? No. seen={0, 1, 2, 3}

Returns [] ← Expected ["a"]!

**Bug:** `enumerate()` yields `(index, item)` but the code unpacks as `(item, index)` — the variables are swapped. The code is checking if integer indices are duplicates (they never are) instead of checking the actual list items.""",
        "fix": """```python
def find_duplicates(lst):
    seen = set()
    duplicates = []
    for index, item in enumerate(lst):
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return duplicates
```

Swap to `for index, item in enumerate(lst)`. Or since the index isn't used, just `for item in lst:`.""",
    },
    {
        "name": "recursive_no_return",
        "code": """def gcd(a, b):
    if b == 0:
        return a
    gcd(b, a % b)""",
        "test_input": "gcd(12, 8)",
        "trace": """Let me trace `gcd(12, 8)`:

- a=12, b=8: b != 0 → call gcd(8, 12 % 8) = gcd(8, 4)
  - a=8, b=4: b != 0 → call gcd(4, 8 % 4) = gcd(4, 0)
    - a=4, b=0: return 4
  - gcd(4, 0) returned 4, but this line is `gcd(b, a % b)` with NO return
  - Returns None implicitly
- Same: the outer call also has no return
- Returns None

Expected 4, got None.

**Bug:** Missing `return` on the recursive call. `gcd(b, a % b)` computes the correct result but discards it because the value isn't returned to the caller.""",
        "fix": """```python
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```

Add `return` before the recursive call.""",
    },
    {
        "name": "min_stack_stale_min",
        "code": """class MinStack:
    def __init__(self):
        self.stack = []
        self.min_val = float('inf')

    def push(self, val):
        self.stack.append(val)
        if val < self.min_val:
            self.min_val = val

    def pop(self):
        return self.stack.pop()

    def get_min(self):
        return self.min_val""",
        "test_input": "push(3), push(1), push(2), pop(), get_min()",
        "trace": """Let me trace:

- push(3): stack=[3], min_val=3
- push(1): stack=[3,1], min_val=1
- push(2): stack=[3,1,2], min_val=1
- pop(): removes 2, stack=[3,1], min_val=1 ✓
- pop(): removes 1, stack=[3], min_val=1 ← WRONG!
- get_min(): returns 1, but the stack only contains [3]. Min should be 3.

**Bug:** When we pop the element that is the current minimum, `min_val` still holds the old minimum. There's no way to recover the previous minimum because we didn't track the history of minimums.""",
        "fix": """```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val

    def get_min(self):
        return self.min_stack[-1]
```

Use a secondary stack to track the history of minimums. When popping a value equal to the current min, also pop from the min stack to reveal the previous min.""",
    },
]

CODE_TRACING_EXAMPLES = [
    {
        "code": """def mystery(s):
    result = {}
    for char in s:
        result[char] = result.get(char, 0) + 1
    return {k: v for k, v in result.items() if v > 1}""",
        "input_expr": 'mystery("abracadabra")',
        "trace": """Let me trace through `mystery("abracadabra")`:

Building the frequency dict character by character:
- 'a': result.get('a', 0)+1 = 1, then 2, 3, 4, 5 → result['a'] = 5
- 'b': appears at positions 1, 8 → result['b'] = 2
- 'r': appears at positions 2, 9 → result['r'] = 2
- 'c': appears at position 4 → result['c'] = 1
- 'd': appears at position 6 → result['d'] = 1

Full result dict: {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}

Then the dict comprehension filters to only entries where v > 1:
{'a': 5, 'b': 2, 'r': 2}

**This function finds all characters that appear more than once in the string, along with their counts.**

Returns: `{'a': 5, 'b': 2, 'r': 2}`""",
    },
    {
        "code": """def transform(data):
    return list(map(lambda x: x[0].upper() + x[1:], filter(lambda x: len(x) > 2, data)))""",
        "input_expr": 'transform(["hi", "hello", "go", "world", "a", "python"])',
        "trace": """Let me break down this nested functional expression:

Step 1 — filter: keep strings with length > 2
- "hi" (len=2) → filtered out
- "hello" (len=5) → kept
- "go" (len=2) → filtered out
- "world" (len=5) → kept
- "a" (len=1) → filtered out
- "python" (len=6) → kept

After filter: ["hello", "world", "python"]

Step 2 — map: capitalize first letter (x[0].upper() + x[1:])
- "hello" → "H" + "ello" = "Hello"
- "world" → "W" + "orld" = "World"
- "python" → "P" + "ython" = "Python"

**This function filters strings shorter than 3 characters, then capitalizes the remaining ones.**

Returns: `["Hello", "World", "Python"]`""",
    },
    {
        "code": """from functools import reduce

def compute(nums):
    evens = [x for x in nums if x % 2 == 0]
    squares = [x ** 2 for x in evens]
    total = reduce(lambda a, b: a + b, squares, 0)
    return total""",
        "input_expr": "compute([1, 2, 3, 4, 5, 6])",
        "trace": """Let me trace step by step:

Step 1 — Filter evens: [x for x in [1,2,3,4,5,6] if x % 2 == 0]
- 1 % 2 = 1, skip
- 2 % 2 = 0, keep
- 3 % 2 = 1, skip
- 4 % 2 = 0, keep
- 5 % 2 = 1, skip
- 6 % 2 = 0, keep
evens = [2, 4, 6]

Step 2 — Square each: [x**2 for x in [2, 4, 6]]
squares = [4, 16, 36]

Step 3 — Sum with reduce:
- reduce starts with accumulator = 0
- 0 + 4 = 4
- 4 + 16 = 20
- 20 + 36 = 56

**This function sums the squares of all even numbers in the input.**

Returns: `56`""",
    },
    {
        "code": """def process(text, n=3):
    words = text.split()
    chunks = [words[i:i+n] for i in range(0, len(words), n)]
    return [' '.join(reversed(chunk)) for chunk in chunks]""",
        "input_expr": 'process("the quick brown fox jumps over the lazy dog")',
        "trace": """Let me trace:

Step 1 — Split: words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
len(words) = 9

Step 2 — Chunk into groups of 3:
- i=0: words[0:3] = ["the", "quick", "brown"]
- i=3: words[3:6] = ["fox", "jumps", "over"]
- i=6: words[6:9] = ["the", "lazy", "dog"]

chunks = [["the","quick","brown"], ["fox","jumps","over"], ["the","lazy","dog"]]

Step 3 — Reverse each chunk and join:
- ["the","quick","brown"] → reversed → ["brown","quick","the"] → "brown quick the"
- ["fox","jumps","over"] → reversed → ["over","jumps","fox"] → "over jumps fox"
- ["the","lazy","dog"] → reversed → ["dog","lazy","the"] → "dog lazy the"

**This function splits text into groups of n words and reverses the order within each group.**

Returns: `["brown quick the", "over jumps fox", "dog lazy the"]`""",
    },
    {
        "code": """def build(pairs):
    result = {}
    for key, value in pairs:
        result.setdefault(key, []).append(value)
    return result""",
        "input_expr": 'build([("a", 1), ("b", 2), ("a", 3), ("b", 4), ("a", 5)])',
        "trace": """Let me trace:

- ("a", 1): "a" not in result → setdefault creates result["a"] = [] → append 1 → result = {"a": [1]}
- ("b", 2): "b" not in result → setdefault creates result["b"] = [] → append 2 → result = {"a": [1], "b": [2]}
- ("a", 3): "a" already in result → setdefault returns existing [1] → append 3 → result = {"a": [1, 3], "b": [2]}
- ("b", 4): "b" already in result → setdefault returns existing [2] → append 4 → result = {"a": [1, 3], "b": [2, 4]}
- ("a", 5): append 5 to existing list → result = {"a": [1, 3, 5], "b": [2, 4]}

**This function groups values by key — it builds a dict where each key maps to a list of all values associated with that key.**

Returns: `{"a": [1, 3, 5], "b": [2, 4]}`""",
    },
    {
        "code": """def interleave(*iterables):
    iterators = [iter(it) for it in iterables]
    result = []
    while iterators:
        next_iterators = []
        for it in iterators:
            try:
                result.append(next(it))
                next_iterators.append(it)
            except StopIteration:
                pass
        iterators = next_iterators
    return result""",
        "input_expr": "interleave([1, 2, 3], ['a', 'b'], [10, 20, 30, 40])",
        "trace": """Let me trace round by round:

Round 1: Take one from each iterator
- iter([1,2,3]) → 1, iter(['a','b']) → 'a', iter([10,20,30,40]) → 10
- result = [1, 'a', 10], all 3 iterators active

Round 2:
- → 2, → 'b', → 20
- result = [1, 'a', 10, 2, 'b', 20], all 3 active

Round 3:
- → 3, → StopIteration (iterator 2 exhausted, dropped), → 30
- result = [1, 'a', 10, 2, 'b', 20, 3, 30], 2 iterators active

Round 4:
- → StopIteration (iterator 1 exhausted), → 40
- result = [1, 'a', 10, 2, 'b', 20, 3, 30, 40], 1 iterator active

Round 5:
- → StopIteration (iterator 3 exhausted)
- iterators = [], loop exits

**This function interleaves multiple sequences round-robin, handling sequences of different lengths gracefully.**

Returns: `[1, 'a', 10, 2, 'b', 20, 3, 30, 40]`""",
    },
]


def generate_debug_example(bug: dict, prompt_idx: int = 0) -> dict:
    template = PROMPT_TEMPLATES_DEBUG[prompt_idx % len(PROMPT_TEMPLATES_DEBUG)]
    prompt = template.format(code=bug["code"])
    response = f"{bug['trace']}\n\n**Fix:**\n\n{bug['fix']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def generate_trace_example(trace: dict, prompt_idx: int = 0) -> dict:
    template = PROMPT_TEMPLATES_TRACE[prompt_idx % len(PROMPT_TEMPLATES_TRACE)]
    prompt = template.format(code=trace["code"], input_expr=trace["input_expr"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": trace["trace"]},
        ]
    }


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic/cot_examples.jsonl")
    parser.add_argument("--prompt-variations", type=int, default=3,
                        help="Number of prompt variations per bug pattern")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    examples = []
    seen = set()

    for bug in BUG_PATTERNS:
        for i in range(args.prompt_variations):
            ex = generate_debug_example(bug, i)
            h = content_hash(ex["messages"][1]["content"])
            if h not in seen:
                examples.append(ex)
                seen.add(h)

    for trace in CODE_TRACING_EXAMPLES:
        for i in range(min(args.prompt_variations, len(PROMPT_TEMPLATES_TRACE))):
            ex = generate_trace_example(trace, i)
            h = content_hash(ex["messages"][1]["content"])
            if h not in seen:
                examples.append(ex)
                seen.add(h)

    random.shuffle(examples)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} chain-of-thought examples")
    print(f"  Debugging examples: {len(BUG_PATTERNS) * args.prompt_variations}")
    print(f"  Code tracing examples: {len(CODE_TRACING_EXAMPLES) * min(args.prompt_variations, len(PROMPT_TEMPLATES_TRACE))}")
    print(f"  After dedup: {len(examples)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
