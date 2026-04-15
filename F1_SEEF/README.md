# Stabilized Ellipsoid Embedding Function Visualizer — F₁

Interactive visualizer for the conjectural stabilized ellipsoid embedding function c_{a,b}(x) of the Hirzebruch surface F₁.

## Quick start

```bash
python3 launch.py
```

This starts a local HTTP server and opens the visualizer in your browser. (A server is needed because the page loads `sesqui_data.json` via fetch.)

Alternatively, from this directory:

```bash
python3 -m http.server 8001
```

then open http://localhost:8001/index.html.

## What it shows

The function c_{a,b}(x) is the infimal c such that the stabilized ellipsoid E(1,x) × ℝ² symplectically embeds into F₁(ca, cb) × ℝ², where F₁(a,b) denotes F₁ with symplectic form giving area a to the line class [L] and area b to the exceptional class [E].

Each sesquicuspidal curve C in homology class d[L] − m[E] with tangency data (p,q) gives the lower bound:

```
c_{a,b}(p/q) >= p / (ad − bm)
```

The conjecture is that c_{a,b}(x) is the smallest continuous nondecreasing function with c(x)/x nonincreasing that satisfies all these bounds. Equivalently, c(x) is the pointwise supremum of the piecewise-linear functions min(L · x / x₀, L), where x₀ = p/q and L = p/(ad − bm), over all sesquicurves.

## Controls

- **a**: symplectic area of the line class [L] (default 1)
- **b**: symplectic area of the exceptional class [E] (must satisfy 0 ≤ b < a)
- **x max**: right edge of the x-axis
- **Volume bound**: overlay sqrt(x / (a² − b²)), the lower bound from symplectic volume
- **Show constraint lines**: draw the individual piecewise-linear bound from each curve
- **Hover**: displays x, c(x), and the binding curve's (p,q), (d,m), delta, and area

## Data

`sesqui_data.json` contains 25,078 curves (3,044 unicuspidal + 22,034 sesquicuspidal) as a JSON array. Each entry is `[p, q, d, m, delta]`.

The data was generated from scattering diagram computations for F₁ with incoming walls (1,0), (0,−1), (−1,−2):

- Unicuspidal curves (delta = 0) computed via BFS of the scattering mutation graph
- Sesquicuspidal curves (delta > 0) enumerated as primitive tuples up to order 100 with coprime (p,q), filtered by nonnegative intersection with all unicuspidal curves

The homology class (d,m) is computed from the alpha tuple via:

```
d = max(a1, a3) + min(p, q)
m = max(a1, a3) - a2
```

where (a1, a2, a3) are the exponents of (t1, t2, t3) in the scattering diagram.

## Files

- `index.html` — self-contained visualizer (HTML + JS + CSS)
- `sesqui_data.json` — curve data
- `launch.py` — convenience launcher
- `README.md` — this file

## Regenerating the data

From the parent directory (`sesqui_and_seef_conjectures/`):

```bash
python3 generate_large_lists.py
```

then copy `sesqui_data.json` into this folder (the generate script also writes the markdown curve lists).
