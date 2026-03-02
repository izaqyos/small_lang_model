export const tokenizerExamples = [
  {
    input: 'def fibonacci(n):',
    tokens: ['def', 'Ġfib', 'onacci', '(', 'n', '):'],
    ids: [142, 3891, 7234, 11, 52, 284],
  },
  {
    input: 'for i in range(10):',
    tokens: ['for', 'Ġi', 'Ġin', 'Ġrange', '(', '10', '):'],
    ids: [263, 187, 199, 1208, 11, 340, 284],
  },
  {
    input: 'class MyModel(nn.Module):',
    tokens: ['class', 'ĠMy', 'Model', '(', 'nn', '.', 'Module', '):'],
    ids: [467, 1502, 3201, 11, 1887, 16, 4502, 284],
  },
];

export const model50MOutputs = {
  fibonacci: `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(
        n - 2)

def main():
    for i in range(10):
        print(fibonacci(i))`,

  sort: `def sort_list(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i] < arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr`,
};

export const model500MOutputs = {
  fibonacci: `def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number.
    
    Args:
        n: Non-negative integer index.
    
    Returns:
        The nth Fibonacci number.
    
    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`,

  sort: `def sort_list(arr: list[int]) -> list[int]:
    """Sort a list using merge sort (O(n log n)).
    
    Args:
        arr: List of comparable elements.
    
    Returns:
        New sorted list.
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = sort_list(arr[:mid])
    right = sort_list(arr[mid:])
    return _merge(left, right)

def _merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result`,
};
