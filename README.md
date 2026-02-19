![](https://github.com/neuralgcm/neuralgcm/raw/main/docs/_static/neuralgcm-logo-light.png)

# Neural General Circulation Models for Weather and Climate

NeuralGCM is a Python library for building hybrid ML/physics atmospheric models
for weather and climate simulation.

- **[Paper](https://arxiv.org/abs/2311.07222)**
- **[Documentation](https://neuralgcm.readthedocs.io/)**
- **License**:
    - Code: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
    - Trained model weights: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

# DYNAMICAL-CORE.md

A detailed code review of the Dinosaur dynamical core used by NeuralGCM.

Source: https://github.com/neuralgcm/dinosaur

---

## 1. Overview & Complexity Level

Dinosaur is a **spectral transform method** atmospheric dynamical core written entirely in JAX. It solves the hydrostatic **primitive equations** on the sphere — the standard equation set for global atmospheric models (vorticity, divergence, temperature, surface pressure, tracers). It also includes a multi-layer **shallow water** solver.

In terms of complexity, this is a **research-grade dry/moist dynamical core**, comparable in equation formulation to the spectral cores of SPEEDY, the original ECMWF IFS, or GFDL's spectral models. It is deliberately simpler than operational NWP dycores (no semi-Lagrangian advection, no non-hydrostatic extension), but it is a complete and physically correct implementation of the spectral primitive equations with semi-implicit time integration.

### Key files

| File | Lines | Purpose |
|------|-------|---------|
| `primitive_equations.py` | ~2100 | Full primitive equations (sigma + hybrid) |
| `spherical_harmonic.py` | ~1100 | Spherical harmonic transforms & Grid |
| `sigma_coordinates.py` | ~450 | Pure sigma vertical coordinate |
| `hybrid_coordinates.py` | ~600 | Hybrid sigma-pressure vertical coordinate |
| `time_integration.py` | ~740 | IMEX time integrators |
| `filtering.py` | ~100 | Spectral filtering for stability |
| `associated_legendre.py` | ~100 | Associated Legendre function evaluation |
| `fourier.py` | - | Fourier basis and derivatives |
| `held_suarez.py` | ~290 | Held-Suarez test case forcing |
| `radiation.py` | ~200 | Top-of-atmosphere solar radiation |
| `vertical_interpolation.py` | ~200 | Sigma ↔ pressure level regridding |
| `units.py` | ~200 | Physical constants and non-dimensionalization |
| `shallow_water.py` | ~250 | Multi-layer shallow water equations |

---

## 2. Prognostic Variables

The `primitive_equations.State` carries:

```python
@tree_math.struct
class State:
    vorticity: Array                    # [layers, m, l] — modal
    divergence: Array                   # [layers, m, l] — modal
    temperature_variation: Array        # [layers, m, l] — modal (T - T_ref)
    log_surface_pressure: Array         # [1, m, l]      — modal
    tracers: Mapping[str, Array]        # e.g. specific_humidity, cloud_water
    sim_time: float | None
```

All prognostic fields are stored in **spectral (modal) space**. Temperature is stored as a **deviation from a reference profile** `T_ref(z)`, which is critical for the semi-implicit scheme's accuracy (the implicit terms linearize around `T_ref`).

---

## 3. Spectral Method — Spherical Harmonics

### 3.1 Transform implementation

The spectral transform decomposes fields into **real spherical harmonic coefficients** Y_l^m, with longitudinal wavenumber `m` and total wavenumber `l`. The transform is a two-step process:

1. **Fourier transform** in longitude (λ): Uses explicit matrix multiplication (not FFT) because the resolutions used (up to ~TL1279) make matrix multiplication competitive, especially on TPUs/GPUs
2. **Associated Legendre transform** in latitude: Matrix multiplication against precomputed normalized associated Legendre polynomials P_l^m(sin θ)

Two implementations exist:
- `RealSphericalHarmonics` — pedagogical, float32 real-valued modal representation with shape `[2M-1, L]`
- `FastSphericalHarmonics` — optimized for performance, supports SPMD model parallelism (sharding across `x`, `y`, `z` mesh axes), uses `tensorfloat32` precision by default, pads arrays to multiples that align with shard sizes

### 3.2 Latitude grid

Supports three latitude node spacings:
- **Gauss-Legendre** (default) — optimal for quadrature, standard in spectral GCMs
- **Equiangular** — equally spaced in latitude
- **Equiangular with poles** — including pole points

### 3.3 Differential operators (all exact in spectral space)

The `Grid` class provides:
- `laplacian(x)` — exact: multiply by eigenvalues `-l(l+1)/a²`
- `inverse_laplacian(x)` — exact: divide by eigenvalues
- `d_dlon(x)` — longitudinal derivative via Fourier basis derivative
- `cos_lat_d_dlat(x)` — latitudinal derivative via associated Legendre recurrence
- `div_cos_lat(u, v)` — divergence of `(u cos θ, v cos θ)`
- `curl_cos_lat(u, v)` — curl (for vorticity)
- `cos_lat_grad(x)` — gradient of scalar field
- `clip_wavenumbers(x, n=1)` — zero out highest `n` total wavenumbers

Velocities are recovered from vorticity/divergence via streamfunction/velocity potential:
```
ψ = ∇⁻²ζ,  χ = ∇⁻²δ
u cos θ = -∂ψ/∂θ + ∂χ/∂λ/(cos θ)
v cos θ = ∂ψ/∂λ/(cos θ) + ∂χ/∂θ
```
This is implemented in `spherical_harmonic.get_cos_lat_vector()`.

---

## 4. Dealiasing

Dealiasing is handled through the **truncation rule** used when constructing the grid. The `Grid.with_wavenumbers()` factory supports:

```python
order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
longitude_nodes = order * longitude_wavenumbers + 1
```

- **Quadratic dealiasing** (default, `order=3`): The "3/2 rule" — nodal grid has 3× the modes to eliminate aliasing from quadratic nonlinear products. This is the standard for spectral GCMs. Example: T85 has wavenumbers up to 85 and uses 64 Gaussian latitudes (128 latitude nodes), 256 longitude nodes.
- **Linear** (`order=2`): For semi-Lagrangian schemes (like TL grids at ECMWF) where horizontal advection is handled in a way that avoids quadratic products
- **Cubic** (`order=4`): Extra dealiasing headroom

The code also provides many **named standard grids**: T21, T31, T42, T85, T106, T170, T213, T340, T425 (quadratic truncation) and TL31 through TL1279 (linear truncation). These match ECMWF naming conventions.

**Additional dealiasing**: The highest total wavenumber is clipped after every evaluation of explicit tendencies (`clip_wavenumbers` at the end of `explicit_terms`), matching the approach used in SPEEDY. This prevents energy buildup at the smallest resolved scales.

---

## 5. Vertical Coordinate Systems

### 5.1 Sigma coordinates (`sigma_coordinates.py`)

Pure terrain-following `σ = p/p_s` coordinates. The `SigmaCoordinates` class stores boundary values (0 to 1) and derives:
- `centers` — midpoints of each layer
- `layer_thickness` — Δσ for each layer
- `center_to_center` — distance between adjacent centers

Key operations:
- `cumulative_sigma_integral` — vertical integration (midpoint rule)
- `centered_vertical_advection` — 2nd-order centered difference vertical advection
- `centered_difference` — ∂x/∂σ

The number of layers is **fully configurable** — set by the length of the boundaries array. NeuralGCM published models use **32 sigma levels** (see the paper), but the code supports any number.

### 5.2 Hybrid sigma-pressure coordinates (`hybrid_coordinates.py`)

The `HybridCoordinates` class implements the standard hybrid formulation used by ECMWF and most operational centers:

```
p(k) = a(k) + b(k) × p_s
```

where `a` (pressure component) and `b` (sigma component) blend from pure pressure levels aloft to terrain-following near the surface. This means:
- **Near the surface**: levels follow terrain (σ-like, b ≈ 1)
- **In the upper atmosphere**: levels are flat pressure surfaces (isobaric, b → 0)

The class provides:
- Factory methods for **ECMWF 137 levels** and **NOAA UFS 127 levels** (loaded from CSV data files)
- `ecmwf137_interpolated(n_levels)` — interpolation from ECMWF 137 to arbitrary count
- `analytic_levels()` — analytic power-law hybrid coordinate generation
- `analytic_ecmwf_like()` — empirically-tuned to concentrate resolution near TOA and surface
- `from_sigma_levels()` — wrap sigma coordinates into hybrid format
- `to_approx_sigma_coords()` — convert back to approximate sigma levels

### 5.3 Is it standard isobaric?

**No.** It is NOT a pure isobaric (pressure) coordinate system. It uses terrain-following coordinates (sigma or hybrid), which is the standard approach for atmospheric GCMs that need to handle topography. Pure isobaric coordinates would intersect the ground and cannot represent topography. The hybrid approach is the same one used by ECMWF IFS, GFDL AM4, CESM/CAM, and essentially all modern GCMs.

---

## 6. Topography / Orography

Topography is represented in **modal (spectral) space** as part of the primitive equations. The surface geopotential enters the divergence equation through:

```
d(divergence)/dt = ... - ∇²(g × orography) - ∇²(Φ') - ∇²(R T_ref ln(p_s))
```

In Dinosaur's `PrimitiveEquationsBase`:
```python
def orography_tendency(self) -> Array:
    return -self.physics_specs.g * self.coords.horizontal.laplacian(self.orography)
```

### How well does it handle topography?

**Limitations of the spectral approach to topography:**

1. **Gibbs phenomenon**: Sharp topographic features (Himalayas, Andes) create spectral ringing. This is mitigated by:
   - Spectral filtering of the orography before use (`truncated_modal_orography`, `filtered_modal_orography`)
   - Exponential filters on the state that damp high-wavenumber oscillations
   - The ability to clip highest wavenumber from orography

2. **Terrain-following coordinate distortion**: In sigma coordinates, coordinate surfaces follow the terrain. Over steep mountains, this creates large pressure gradient errors because the horizontal pressure gradient force must be computed as a small difference of large terms. Hybrid coordinates alleviate this by transitioning to flat pressure surfaces aloft.

3. **Resolution dependence**: At T85 (~1.4° resolution), major mountain ranges are resolved but local peaks and valleys are smoothed. Higher resolutions (T170, T340) progressively resolve finer topographic features.

NeuralGCM provides several orography initialization options:
- `ClippedOrography` — modal orography with highest wavenumbers clipped
- `FilteredCustomOrography` — orography from external data with spatial filtering
- `LearnedOrography` — orography with a **learnable correction** (neural network adjusts the spectral representation)
- `ModalOrographyWithCorrection` (experimental) — adds a learnable modal correction

The learnable orography correction is notable — it allows the model to adjust the effective topography to compensate for the smoothing inherent in spectral representation.

---

## 7. Primitive Equations Implementation

### 7.1 Equation structure

The equations follow the vorticity-divergence formulation standard in spectral GCMs, based on Durran, "Numerical Methods for Fluid Dynamics" §8.6, and Simmons & Burridge (1981) for hybrid coordinates.

**Explicit terms** (nonlinear, computed in nodal space then transformed to modal):
- Vorticity advection: `-k · ∇ × ((ζ + f)(k × v) + σ̇ ∂v/∂σ + RT'∇(ln p_s))`
- Divergence advection + kinetic energy: same curl → div, plus `-∇²(KE)` and orography
- Temperature: horizontal advection + vertical advection + adiabatic heating (κ T ω/p)
- Surface pressure: `-Σ(u·∇(ln p_s)) Δσ`
- Tracers: horizontal + vertical advection

**Implicit terms** (linear, fast gravity waves — treated implicitly for stability):
- Divergence: `-∇²(G·T' + R·T_ref·ln(p_s))` where G is the geopotential weight matrix
- Temperature: `H·divergence` where H encodes adiabatic heating from reference temperature
- Surface pressure: `-Δσ · divergence`
- Vorticity: zero (no implicit vorticity tendency)

This is the standard IMEX (implicit-explicit) splitting used in all semi-implicit spectral GCMs.

### 7.2 Diagnostic state computation

Before evaluating explicit terms, a `DiagnosticState` is computed from the modal state:
1. Transform vorticity, divergence, temperature, tracers to nodal space
2. Recover `cos(θ)·(u, v)` from vorticity/divergence via streamfunction/velocity potential
3. Compute `∇(ln p_s)` and `u·∇(ln p_s)`
4. Compute σ̇ (vertical velocity) from the continuity equation
5. Compute vertical advection terms

### 7.3 Implicit inverse

The semi-implicit scheme requires solving `(I - η·L)⁻¹ x` where L is the implicit operator. This is a coupled system in (divergence, temperature, log_surface_pressure) at each wavenumber `l`.

The implicit matrix has block structure per wavenumber:
```
[  I    ηλG   ηRλT_ref ]
[ ηH     I      0      ]
[ ηΔσ    0      1      ]
```

Three solution methods are provided:
- **`split`** (default): Precompute full inverse in float64, apply as 9 matrix-vector products
- **`stacked`**: Concatenate variables, apply single inverse (pedagogical)
- **`blockwise`**: Block-wise Schur complement inversion — fewer matrix-vector products, better for vertical sharding

The implicit matrix is computed in **float64** (NumPy, not JAX) and the inverse is precomputed at initialization, not during the JIT-compiled forward pass. This is possible because the implicit operator depends only on the grid, reference temperature, and physical constants — all static.

### 7.4 Moisture coupling

The primitive equations support moisture in three progressive levels:
- `PrimitiveEquationsSigma` / `PrimitiveEquationsHybrid` — dry dynamics
- With `humidity_key` set — moist dynamics with virtual temperature correction. Specific humidity modifies:
  - Geopotential via virtual temperature: `T_v = T(1 + (R_v/R - 1)q)`
  - Additional divergence/vorticity tendencies from moisture pressure gradient terms
  - Modified adiabatic heating with moist heat capacity
- With `cloud_keys` set — cloud condensate effects on virtual temperature

---

## 8. Time Integration

### 8.1 Available integrators

All IMEX (Implicit-Explicit) schemes that solve `∂x/∂t = F(x) + G(x)`:

| Scheme | Order (explicit/implicit) | Function |
|--------|--------------------------|----------|
| Backward-Forward Euler | 1/1 | `backward_forward_euler` |
| Crank-Nicolson + Heun RK2 | 2/2 | `crank_nicolson_rk2` |
| Crank-Nicolson + Williamson RK3 | 3/2 | `crank_nicolson_rk3` |
| Crank-Nicolson + Carpenter-Kennedy RK4 | 4/2 | `crank_nicolson_rk4` |
| IMEX-RK SIL3 (Whitaker & Kar 2013) | 3/2 | `imex_rk_sil3` |
| Semi-implicit leapfrog | 2/2 | `semi_implicit_leapfrog` |
| Forward Euler (explicit only) | 1 | In NeuralGCM `time_integrators.py` |
| RK4 (explicit only) | 4 | In NeuralGCM `time_integrators.py` |

Plus a semi-Lagrangian vertical advection option (`semi_lagrangian_vertical_advection_step_sigma`) as an alternative to Eulerian centered-difference vertical advection.

### 8.2 Filters applied between time steps

- **Exponential filter**: Attenuates high total wavenumber modes by `exp(-a((k-c)/(1-c))^(2p))`. Default parameters: attenuation=16, order=18, cutoff=0. References Hou & Li (2007) and Gottlieb & Shu (1997) for Gibbs phenomenon resolution.
- **Horizontal diffusion filter**: Scale-selective damping proportional to `(-λ_l)^n` where λ_l is the Laplacian eigenvalue
- **Robert-Asselin filter**: For leapfrog time stepping, smooths the computational mode
- **Global mean preservation filter**: Fixes global mean of specified fields after each step
- **Digital filter initialization** (DFI): Lanczos-windowed initialization to remove fast gravity waves (Lynch & Huang 1992)

### 8.3 Nested checkpoint scan

A notable utility: `nested_checkpoint_scan` provides **memory-efficient gradient computation** for long trajectories. By nesting `jax.lax.scan` with `jax.checkpoint`, it reduces memory from O(T) to O(max(nested_lengths)) at the cost of recomputation. This is critical for training NeuralGCM on long sequences.

---

## 9. Ocean Surface Forcing & External Forcing

### 9.1 What the dynamical core provides

The Dinosaur dynamical core itself has **no built-in ocean surface forcing, boundary layer, convection, or radiation schemes** (beyond the Held-Suarez test case and TOA solar radiation). These are all handled as **external forcing terms** composed with the dynamical core equations via `time_integration.compose_equations()`.

### 9.2 Held-Suarez test case

The only built-in "physics" is the Held-Suarez (1994) test case for dynamical core intercomparison:
- **Newtonian cooling**: temperature relaxation toward a zonally symmetric equilibrium profile
- **Rayleigh friction**: momentum damping in the planetary boundary layer (σ > σ_b = 0.7)
- Available for both sigma (`HeldSuarezForcingSigma`) and hybrid (`HeldSuarezForcingHybrid`) coordinates

### 9.3 Solar radiation

`radiation.py` provides top-of-atmosphere incident solar radiation (`OrbitalTime` + declination/hour-angle calculation). This gives the diurnal and seasonal cycles of insolation but does NOT include atmospheric radiative transfer.

### 9.4 How NeuralGCM provides forcing

In NeuralGCM, the gap between the minimal dycore physics and a full GCM is filled by:
- **Neural network parameterizations** (`parameterizations.py`) that learn subgrid tendencies for temperature, moisture, and momentum
- **Dynamic data forcing** (`forcings.py`) that interpolates time-varying external data (e.g., SST, sea ice) into the model at each timestep
- **Correctors** and **fixers** that apply physical constraints (e.g., mass/energy conservation)

The SST/ocean forcing comes in as **prescribed boundary conditions** through the forcing data mechanism — not from a coupled ocean model. SST fields are loaded from ERA5 or similar reanalysis data and interpolated to the model time step.

---

## 10. Differentiability

This is a critical design aspect. Every component of Dinosaur is written to be **fully differentiable through JAX's autodiff system**.

### 10.1 What makes it differentiable

1. **Pure JAX implementation**: All computations use `jax.numpy` and JAX-compatible operations. No external Fortran/C libraries or non-differentiable numerical routines.

2. **tree_math integration**: The `@tree_math.struct` decorator on `State` and diagnostic structs makes them behave like vectors under `+`, `*`, scalar multiplication. This enables `tree_math.unwrap/wrap` to apply JAX transformations (jit, grad, vmap) to entire state-based functions.

3. **Static implicit inverse**: The implicit matrix inverse is precomputed in NumPy (float64) and stored as a static array. During the JIT-compiled forward pass, only matrix-vector products are performed — these are standard `jnp.einsum` operations that are fully differentiable. The step size `η` must be a concrete (non-traced) value, not a JAX tracer.

4. **No branching on dynamic values**: Control flow like if/else is used only on static values (grid parameters, configuration). Dynamic state values never appear in Python control flow, which would break JIT compilation and differentiation.

5. **Precision management**: All `einsum` operations use `precision=jax.lax.Precision.HIGHEST` to maximize numerical accuracy. The implicit inverse uses float64 NumPy. The spherical harmonic transforms use `tensorfloat32` by default for performance but this is configurable.

### 10.2 How NeuralGCM uses differentiability

The full differentiability enables:
- **End-to-end training**: Gradients flow from a forecast loss backward through the time integrator, through the spectral transforms, through the primitive equations, and into the neural network parameters
- **`jax.lax.scan`-based rollouts**: Multi-step forecasts are generated with `scan`, and `nested_checkpoint_scan` enables gradient checkpointing to control memory
- **Vectorization with `vmap`**: Ensemble forecasts or batch training over multiple initial conditions

### 10.3 Potential differentiability concerns

1. **`jnp.interp` / `searchsorted`**: Used in vertical interpolation. These have approximate gradients (piecewise linear), which is fine for training but not exact.

2. **`jnp.maximum` / `jnp.minimum`**: Used in moisture clipping and geopotential safety bounds. Gradients are zero on one side — this can cause gradient issues if the model frequently hits these bounds, but it's standard practice.

3. **Spectral transform precision**: The `tensorfloat32` default for `FastSphericalHarmonics` trades some accuracy for speed. For gradient-sensitive applications, full `float32` or `highest` precision can be selected.

4. **No explicit energy conservation enforcement in the dycore**: The spectral method conserves energy to the extent that the time integration scheme does, but there is no explicit Hamiltonian structure or symplectic integrator. NeuralGCM adds separate energy/mass conservation fixers.

---

## 11. SPMD Parallelism

The dynamical core has built-in support for distributed computation:

- `CoordinateSystem` carries a 3D `spmd_mesh` with axes `('z', 'x', 'y')` corresponding to vertical levels, longitude, and latitude
- Two sharding modes:
  - **Dycore sharding** `P('z', 'x', 'y')` — for spectral transforms and equation evaluations
  - **Physics sharding** `P(None, ('x', 'z'), 'y')` — for column-wise operations (neural nets, vertical interpolation), merging the vertical axis into a spatial axis
- `FastSphericalHarmonics` uses `shard_map` for parallel Legendre and Fourier transforms
- The code handles padding to align array dimensions with shard sizes

---

## 12. Non-Dimensionalization

All physical quantities are non-dimensionalized using a `Scale` object (from `scales.py`). The default scale normalizes by Earth's radius and angular velocity:

```python
# Default scales
radius → 1.0
angular_velocity → 1.0
# Derived:
time_scale = 1/Ω ≈ 13713 seconds
length_scale = a ≈ 6.37 × 10⁶ m
```

The `SimUnits` class (`units.py`) stores non-dimensional physical constants (g, R, κ, R_vapor, Cp, Cp_vapor) and provides `nondimensionalize()` / 
