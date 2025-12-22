# QUBO‑only LaTeX Mini System

This document describes a **minimal LaTeX‑like language** used to build a **QUBO (Quadratic Unconstrained Binary Optimization)** matrix.

The goal of the system is simple:

> **Given `(x, y)` → return `Q[x][y]`**

All parsing, summations, and expressions ultimately reduce to filling a QUBO matrix.

---

## 1. Core Idea

A QUBO problem is expressed as:

```
min \sum_{i \le j} Q_{ij} q_i q_j
```

Where:

* `q_i ∈ {0,1}` are **qubits (binary variables)**
* `Q` is a **real‑valued symmetric matrix**

This system parses LaTeX‑like math and **builds `Q[x][y]` directly**.

---

## 2. Variables

### Qubits

All binary variables use the letter **`q`**:

```
q_0, q_1, q_2, ...
```

Rules:

* Variable name is **one character**: `q`
* Index is an integer after `_`
* Example: `q_5` → qubit with index `5`

---

## 3. Numbers

Supported numeric literals:

```
1
-3
2.5
```

Numbers are treated as **coefficients** in the polynomial.

---

## 4. Operators

Supported operators:

| Operator | Meaning                     |
| -------- | --------------------------- |
| `+`      | addition                    |
| `-`      | subtraction                 |
| `*`      | multiplication              |
| implicit | multiplication (`q_i q_j`)  |
| `^`      | power (used mainly as `^2`) |
| `=`      | equality (for constraints)  |
| `<`, `>` | index relations             |

Only **quadratic polynomials** are allowed.

---

## 5. Summation (`\sum`)

### Syntax

```
\sum_{i = a}^b <expression>
```

### Meaning

* `_` **creates a loop variable** with a **starting value**
* `^` **increments the variable until the upper bound**

Example:

```
\sum_{i=0}^3 q_i
```

Expands to:

```
q_0 + q_1 + q_2 + q_3
```

---

### Nested sums

```
\sum_{i=0}^2 \sum_{j=i+1}^3 q_i q_j
```

Expansion:

```
q_0 q_1 + q_0 q_2 + q_0 q_3
+ q_1 q_2 + q_1 q_3
+ q_2 q_3
```

---

## 6. Index Relations

Inside summation bounds you may use:

```
i < j
i <= j
```

These restrict loop expansion and are evaluated during sum expansion.

---

## 7. Valid Polynomial Forms

After all expansions, every term must reduce to **one of these**:

| Form        | Meaning         |
| ----------- | --------------- |
| `c`         | constant offset |
| `a q_i`     | linear term     |
| `b q_i q_j` | quadratic term  |

Rules:

* `q_i q_i` is treated as `q_i`
* Degree > 2 is **invalid**

---

## 8. QUBO Matrix Construction

Each valid term updates the matrix:

```
b q_i q_j  →  Q[min(i,j)][max(i,j)] += b
```

Linear term:

```
a q_i → Q[i][i] += a
```

Constant term:

```
c → energy offset
```

The matrix is symmetric by construction.

---

## 9. Example

### Input

```
\sum_{i=0}^2 q_i + 2 q_0 q_2
```

### Result

```
Q[0][0] += 1
Q[1][1] += 1
Q[2][2] += 1
Q[0][2] += 2
```

Now:

```
get_Q(0,2) → 2
get_Q(2,0) → 2
```

---

## 10. Design Constraints (Intentional)

This system **does NOT support**:

* integrals
* fractions (`\frac`)
* functions (`sin`, `log`, …)
* vectors or matrices
* formatting commands

Everything exists for **one purpose only**:

> **Build a QUBO matrix efficiently and deterministically**

---


## 11. Summary

* `q` is the qubit variable
* `_` defines the starting index in `\sum`
* `^` defines the ending index in `\sum`
* All math reduces to `(coefficient, i, j)`
* Final output is a QUBO matrix

This minimal design keeps the system **fast, predictable, and scalable**.
