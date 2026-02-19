Jovian 
---

## 1. Motivation and concept

Deep learning has transformed velocity extraction from scalar imagery in two domains — ocean surface currents from geostationary SST (GOFLOW) and laboratory fluid velocimetry from particle images (RAFT-PIV) — but has never been applied to planetary atmospheric winds. This programme builds JOVE, the planetary analog of GOFLOW, exploiting an even stronger physical basis: in the surface quasi-geostrophic (SQG) limit, JIRAM thermal radiance is directly proportional to the surface buoyancy field, which inverts analytically to a streamfunction. The observable *is* the dynamical variable.

JOVE is tightly coupled to a reduced dynamical modeling framework — a staged hierarchy of QG-class models (BT-QG → SQG → QG+1) built on the same solver, forced by stochastic cloud convection, and calibrated against JOVE-derived velocity fields via simulation-based inference (SBI). The reduced model constrains Jupiter's cloud-layer dynamics quantitatively (effective stratification, convective injection scale, ageostrophic corrections) while simultaneously informing JOVE's training data distribution.

The spherical solver is built on `dinosaur`, the JAX-native dynamical core from NeuralGCM (Kochkov et al. 2024, Nature), which provides production-validated, GPU-optimized, fully differentiable spherical harmonic transforms. We write thin QG/SQG/QG+1 dynamical layers and passive scalar advection on top of this foundation.

## 2. Data

**JIRAM** (4.78 μm, ~1 bar sensing depth, 256×432 pixels, ~12 km/pixel at perijove): Consecutive frames as Juno passes over the poles create overlapping fields of view separated by ~15–30 minutes. Priority perijoves: PJ4 (south polar, best-characterized, published velocity fields from Siegelman+2022, Ingersoll+2022), PJ7 (north polar, Grassi+2018), PJ5/6/20/33 (multi-orbit coverage, Scarica+2022). Source: NASA PDS Atmospheres Node. Processing requires SPICE kernels for geometric calibration and projection to polar stereographic grids at ~12 km resolution.

**JunoCam** (visible, 0.3–0.7 bar, pushbroom): Higher spatial resolution near perijove (~5 km/pixel) but senses a different altitude. Secondary priority; useful for cross-instrument vertical shear validation.

**Johns Hopkins Turbulence Database**: DNS velocity fields (rotating stratified turbulence at 4096³, passive scalar mixing) accessed via Python API. Serves as augmentation data for non-geostrophic small-scale diversity in the training set and as the basis for RAFT-PIV baseline fine-tuning.

## 3. Baseline methods

Four tiers of velocity extraction, all applied to identical projected JIRAM image pairs, all compared using identical metrics.

**Tier 1 — Cross-correlation / CIV.** Interrogation-window-based 2D cross-correlation with sub-pixel Gaussian peak fitting. Also the MAD (L1) variant used by Grassi et al. Target: reproduce Grassi+2018 and Ingersoll+2022 on PJ4.

**Tier 2 — ACCIV.** Advection Corrected CIV (Asay-Davis et al. 2009): iteratively advects images to a common time, enabling tracking over longer temporal separations. Requires semi-Lagrangian advection (RK4 + bicubic interpolation).

**Tier 3 — Variational optical flow.** Per-pixel velocity from minimizing a variational functional. Three variants: Horn-Schunck, transport-equation data term (Liu & Shen 2008), divergence-constrained (Kadri-Harouna et al. 2013). All in JAX.

**Tier 4 — RAFT.** Pre-trained RAFT (Teed & Deng 2020) applied zero-shot, then fine-tuned on (a) JHTDB and (b) our spherical QG/SQG training data. The zero-shot experiment is an informative proof of concept on its own.

## 4. Spherical synthetic data generation

All synthetic training data is generated on a rotating sphere in physical planetary units using the same dinosaur-based solver infrastructure as the reduced modeling programme (Section 7).

### 4.1 Solver: dinosaur + QG/SQG/QG+1 wrappers

Built on `dinosaur` (`github.com/neuralgcm/dinosaur`). The Jacobian J(ψ, q) requires ~6 SHTs per RHS evaluation. Passive scalar advection reuses the same infrastructure at ~2 extra SHTs per scalar. Changing planetary parameters switches between Jupiter and Saturn.

### 4.2 Dynamical configurations

**(a) Barotropic QG on sphere.** Stochastic forcing at injection wavenumber ℓ_inject. Produces zonal jets, polar vortex accumulation, inverse cascade.

**(b) SQG on sphere.** Surface buoyancy b_s maps to streamfunction via parameterized kernel K(ℓ; θ) = −C/(ℓ(ℓ+1) + ℓ_d²)^{p/2}. **The most important configuration**: JIRAM radiance ∝ b_s, and the SQG inversion gives velocity from a single observable. Training data contains the physically correct relationship between observable and velocity. With uniform N and zero interior PV, this is a purely 2D solver.

**(c) QG+1 on sphere** (Du, Smith & Bühler). Same prognostic as SQG (single surface buoyancy sheet), but velocity includes O(Ro) ageostrophic corrections from diagnostic BVP solves. Analytical leading-order vertical structure + small Chebyshev grid (N_z = 16) for correction potentials. Provides balanced divergence and cyclone-anticyclone asymmetry. QG+1 training data may outperform SQG — testable.

**(d) Supplementary.** Random spectral realizations with prescribed E(ℓ), moist-forced variants, JHTDB augmentation.

### 4.3 Passive scalars, patch extraction, and data volume

Multiple passive scalars per simulation (conservative, diffusive, cloud-like). Global fields projected to polar stereographic and cylindrical patches matching JIRAM/JunoCam geometry. ~300,000 training pairs per simulation, ~240 million total from ~3,000 GPU-hours and ~2 TB.

## 5. JOVE model

### 5.1 Architecture and inputs

Inputs: {θ(t₁), θ(t₂)} plus auxiliary channels (latitude, Δt, emission angle). Optional: SQG-inverted velocity as physics-based prior channel. Outputs: {u(x,y), v(x,y)}. Start U-Net; RAFT backbone v2.

### 5.2 Loss function and training

Four loss components: L_vel (supervised), L_spec (spectral matching), L_adv (advective consistency — self-supervised on real data), L_div (divergence penalty). L_adv is critical: it provides signal on real JIRAM data where no velocity ground truth exists. Training: fully supervised → semi-supervised → iterative refinement.

### 5.3 SQG inversion as testable prediction

Single-image SQG-inverted velocities vs cloud-tracked velocities from image pairs. If they correlate: validates SQG prior, strengthens training rationale, standalone GRL letter.

## 6. Validation

**Synthetic**: endpoint error, angular error, E(ℓ) spectrum recovery, vorticity PDF, divergence statistics — all on held-out spherical test data.

**Real**: JOVE vs all four baseline tiers on JIRAM PJ4; comparison with published Siegelman+2022 and Ingersoll+2022; physical consistency (Rossby number, gradient wind, energy flux).

**Extensions**: Cross-instrument JIRAM/JunoCam vertical shear; Saturn with Cassini data.

## 7. Reduced dynamical modeling: BT-QG → SQG → QG+1

### 7.1 Overview

A staged hierarchy of forced-dissipative models on the sphere with tropical quarantine, built on the same dinosaur solver as JOVE training data. Energy source: stochastic cloud convection on a single surface buoyancy sheet (not Eady mean state). Uniform stratification N and zero interior PV make all three stages effectively 2D. Parameters inferred from JOVE-extracted velocity statistics via SBI.

### 7.2 Tropical quarantine

Equatorial region (|φ| ≲ 25–35°) annihilated by smooth sponge relaxation. Full sphere is carried (spherical harmonics are global) but each hemisphere evolves independently. Science region extends as far equatorward as the sponge permits — determined experimentally in Stage 0.

### 7.3 Stages

**Stage 0: Benchmarking.** All three equation sets on dinosaur. Cost vs resolution. Sponge validation. Time integration choice.

**Stage I: BT-QG.** Forced barotropic dynamics. Sponge proof-of-concept, SBI pipeline demonstration.

**Stage II: Pure SQG.** Single surface buoyancy sheet, spectral-diagonal inversion. No barotropic mode (no energy source → decays). Targets: forward energy transfer, frontogenesis, intermittency, polar vortex structure.

**Stage III: QG+1.** O(Ro) corrections via diagnostic BVP solves (Chebyshev, N_z = 16). Incremental: QG+1-0 (≡ SQG) → QG+1-Φ (skewness) → QG+1-full (balanced divergence, target σ_δ/σ_ζ ≈ 0.64). Optional flow-dependent forcing (convection modulated by frontogenesis/convergence).

### 7.4 Critical conventions

Anomaly forcing (F̂_00 = 0), gauge-fix every inversion (ψ̂_00 = 0), solvability constant C_q(t) for QG+1 Neumann solve, masked means for diagnostics only.

### 7.5 JOVE ↔ reduced model coupling

Same solver for both. SQG inversion test validates both programmes. JOVE velocities are SBI "observations". SBI-inferred parameters constrain JOVE training data. QG+1 training data may outperform SQG.

### 7.6 Escalation logic

BT-QG → SQG (needs forward cascade) → QG+1 (needs asymmetry/divergence) → flow-dependent forcing (needs intermittency). Compare: most diagnostics matched with fewest parameters.

## 8. Proposal targets

NASA CDAP (Cassini + Juno data reanalysis) and NSF CDS&E (computational methodology) are the strongest fits. The narrative: GOFLOW's creators partner with the leading JIRAM dynamics expert to transfer simulation-trained deep learning from ocean to planetary atmospheres, built on NeuralGCM's production-validated JAX infrastructure and exploiting a stronger physical basis (SQG inversion) than exists for ocean SST. The reduced modeling framework with SBI provides the first quantitative constraints on Jupiter's cloud-layer dynamics from observed velocity statistics.

