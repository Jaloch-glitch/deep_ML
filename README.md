# Deep-ML Practice

Daily machine learning problem solving focused on understanding fundamentals, not memorizing solutions.

## Philosophy

> "If you can't implement it from scratch, you don't truly understand it."

This repository tracks my journey through [Deep-ML.com](https://www.deep-ml.com/) problems. Each solution prioritizes:

1. **Understanding over speed** - Break down the problem before coding
2. **ML best practices** - Use `zip()`, comprehensions, vectorized operations
3. **Two implementations** - Pure Python first, then NumPy for production

## Problems Solved

| Problem | Core Concept | Pattern |
|---------|-------------|---------|
| Matrix-Vector Multiply | Dot products | `sum(r*v for r,v in zip(row, vec))` |
| Transpose | Column extraction | `zip(*matrix)` |
| Reshape | Flatten then chunk | `flat[i:i+cols]` with `range(0, len, step)` |
| Row/Column Means | Aggregation | `zip(*matrix)` for columns |
| Scalar Multiply | Element-wise ops | Nested comprehension |
| Matrix Transform T⁻¹AS | Change of basis | Invertibility check + chain multiply |
| Eigenvalues | Spectral decomposition | `np.linalg.eig()` |
| 2x2 Inverse | Linear algebra | Swap diagonal, negate off-diagonal, divide by det |

## Key Patterns

### Pure Python

```python
# Dot product
sum(a * b for a, b in zip(vec1, vec2))

# Matrix-vector multiply
[sum(r * v for r, v in zip(row, vector)) for row in matrix]

# Transpose
[list(col) for col in zip(*matrix)]

# Flatten
[elem for row in matrix for elem in row]

# Reshape (flatten then chunk)
[flat[i:i+cols] for i in range(0, len(flat), cols)]
```

### NumPy Sandwich Pattern

```python
def operation(A, ...):
    # 1. Convert to NumPy
    A_array = np.array(A, dtype=float)

    # 2. Do operations (fast, clean)
    result = A_array @ B_array  # or .T, .reshape(), etc.

    # 3. Convert back to list
    return result.tolist()
```

## What I Learned

**On `zip()`**: Pairs elements position-by-position. With `*` unpacking, `zip(*matrix)` extracts columns.

**On comprehensions**: Read outer-to-inner. `[f(x) for row in matrix for x in row]` means "for each row, for each element in that row."

**On reshape**: Always flatten first, then chunk. The slice `[i:i+width]` cuts a ribbon of exactly `width` elements.

**On invertibility**: A matrix is invertible iff `det != 0`. If det = 0, the matrix "crushes" dimensions and information is lost.

## Structure

```
deep-ml/
├── practice.ipynb    # All solutions with explanations
└── README.md         # This file
```

## Running

```bash
# Activate Jupyter environment
source ~/jupyter-env/bin/activate

# Start JupyterLab
jupyter lab practice.ipynb
```

## Resources

- [Deep-ML.com](https://www.deep-ml.com/) - Daily ML problems
- [Karpathy's Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Neural networks from scratch

## Author

Felix Onyango ([@Jaloch-glitch](https://github.com/Jaloch-glitch))
