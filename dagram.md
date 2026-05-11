# PPT++ Diagram Documentation
## How to Modify the Baseline PPT Diagrams for PPT++

This document explains every figure in the original PPT paper and exactly how to modify each one to reflect our PPT++ contributions.

---

## Original PPT Paper Figures (arXiv:2310.01812)

### Figure 1: Motivation — Comparison of Compression Methods
**Original:** 4-column grid showing (a) Original images, (b) Pruning only (grey patches), (c) Pooling only (color-outlined groups), (d) PPT hybrid

**PPT++ Modification → Add column (e):**
- **(e) PPT++ (Ours):** Same hybrid as PPT, but:
  - Pruned patches shown with **diagonal stripes** (not solid grey) → indicates info was squeezed back
  - Pooled groups shown with **spatially contiguous** boundaries → spatial-aware BSM keeps nearby tokens together
  - More nuanced per-sample decisions visible (adaptive threshold)

---

### Figure 2: PPT Architecture Overview (Main Method Diagram)
**Original:** 3 panels (a) Module insertion, (b) Decision logic, (c) Multi-layer visualization

**PPT++ Modifications:**

#### Panel (a) — Module Insertion:
- Label changes: "Adaptive Token Compression (ATC)" → **"PPT++ Adaptive Compression"**
- Add note: "Zero Additional Parameters (same as PPT)"
- The module box stays in the same position (after MSA, before MLP)

#### Panel (b) — Decision Logic (MAJOR CHANGES):
```
ORIGINAL PPT:                         PPT++ (MODIFIED):
┌─────────────────┐                  ┌─────────────────┐
│ Compute Score_i │                  │ Compute Score_i │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
┌────────┴────────┐                  ┌────────┴────────┐
│ S_op = Var(...)  │                  │ S_op = Var(...)  │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
┌────────┴────────┐         ★NEW →  ┌────────────────────────┐
│ S_op > τ ?      │                  │ S_op > τ_l ?           │
│ (global fixed τ)│                  │ τ_l = μ_l + α·σ_l     │
└───┬─────────┬───┘                  │ (★ ADAPTIVE PER-LAYER) │
    │         │                      └───┬─────────────┬─────┘
    │YES      │NO                        │YES          │NO
    ▼         ▼                          ▼             ▼
┌───────┐  ┌───────┐             ┌──────────────┐  ┌──────────────┐
│ PRUNE │  │ POOL  │             │    PRUNE      │  │    POOL      │
│ Top-K │  │  BSM  │             │    Top-K      │  │ ★SPATIAL BSM │
└───────┘  └───────┘             │ ★+ SQUEEZE    │  │ S=α·Sv+(1-α)│
                                 │ y=x+γΣwi·xi  │  │    ·S_sp     │
                                 └──────────────┘  └──────────────┘
```

**Three highlighted boxes with star (★) markers:**
1. Adaptive threshold box (yellow highlight, orange border)
2. Token squeezing added below pruning (red highlight, orange border)
3. Spatial-aware BSM replaces standard BSM (green highlight, orange border)

#### Panel (c) — Multi-layer Visualization:
- Same grid format but add **adaptive decisions** labels at each layer
- Show that different images get different decisions at same layer
- Add threshold values: "τ₃=7.2e-6, τ₆=7.3e-5, τ₉=3.2e-5"

---

### Figure 3: Variance Curves (Key Motivation)
**Original:** 12 sub-plots showing variance scatter per layer with red average

**PPT++ Modification → New Overlay Figure:**
- Keep the same 12 sub-plots
- ADD: **Green dashed line** at each compression layer showing the adaptive τ_l value
- ADD: **Shaded region** between the per-layer τ and the fixed τ to highlight the gap
- ADD: Annotation showing "60× variance range — single τ cannot serve all layers"
- NEW PANEL: Same format for LV-ViT-S (16 sub-plots) showing completely different scale

---

### Figure 7: Layer-by-Layer Image Processing Visualization
**Original:** Grid with rows=images, columns=layers showing pruned (grey) and pooled (color) patches

**PPT++ Modifications:**
- **Pruned patches:** Change from solid grey to **striped grey** (diagonal lines) → indicates token squeezing recovered the info
- **Pooled patches:** Groups should be more **spatially contiguous** (adjacent patches same color) → spatial-aware BSM
- Add **"Squeezed →"** label on pruned patches
- Add **"Spatial BSM"** label on pooled groups
- Show the squeezing "ghost" effect: faint outline around pruned patches showing they contributed to nearby tokens

---

### Figure 9: Threshold τ Sensitivity
**Original:** Two curves (fine-tuned, off-the-shelf) showing accuracy vs. τ

**PPT++ Modification:**
- Keep original curves for reference (lighter, labeled "PPT")
- ADD: Horizontal band showing "Adaptive τ range" across layers
- ADD: Point markers showing per-layer adaptive thresholds
- ADD: Text annotation: "PPT++ eliminates this sensitivity entirely"

---

## NEW Figures for PPT++

### NEW Figure A: Token Squeezing Mechanism (Step-by-Step)
```
Step 0: Score & Rank     Step 1: Top-K Select    Step 2: Match           Step 3: Squeeze
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ T1 ████████████ │     │ R1 ▓▓▓▓▓▓▓▓▓▓  │     │ R1 ◄━━━━━ P1   │     │ R1 + γ·w₁·P1   │
│ T2 ██████████   │     │ R2 ▓▓▓▓▓▓▓▓▓   │     │ R2             │     │ R2              │
│ T3 ████████     │     │ R3 ▓▓▓▓▓▓▓▓    │     │ R3 ◄━━━━━ P2   │     │ R3 + γ·w₂·P2   │
│ T4 ██████       │     │ R4 ▓▓▓▓▓▓▓     │     │ R4             │     │ R4              │
│ ─── threshold ───     │ ────────────────│     │ R5 ◄━━━━━ P3   │     │ R5 + γ·w₃·P3   │
│ T5 ████  (pruned)     │ P1 ░░░░░░░     │     │                │     │                 │
│ T6 ██   (pruned)      │ P2 ░░░░░       │     │ cos-sim match  │     │ ✓ Info recovered│
│ T7 █    (pruned)      │ P3 ░░░         │     │                │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### NEW Figure B: Spatial-Aware BSM Comparison
```
Standard BSM (PPT):                    Spatial-Aware BSM (PPT++):
┌──────────────────────┐              ┌──────────────────────┐
│ ┌─┐ ┌─┐     ┌─┐     │              │ ┌─┬─┐         ┌─┐   │
│ │A│ │A│     │A│     │              │ │A│A│         │B│   │
│ └─┘ └─┘     └─┘     │              │ └─┴─┘         └─┘   │
│                       │              │                       │
│ ┌─┐         ┌─┐ ┌─┐ │              │ ┌─┐     ┌─┬─┬─┐     │
│ │B│         │B│ │B│ │              │ │C│     │B│B│B│     │
│ └─┘         └─┘ └─┘ │              │ └─┘     └─┴─┴─┘     │
│                       │              │                       │
│ Merges distant tokens │              │ Merges nearby tokens  │
│ (same feature, far)   │              │ (spatial + feature)   │
└──────────────────────┘              └──────────────────────┘

S_visual only                          S_fused = α·S_vis + (1-α)·S_spatial
→ Breaks spatial structure             → Preserves spatial coherence
```

### NEW Figure C: Cross-Architecture Comparison (DeiT-S vs LV-ViT-S)
```
DeiT-S (12 blocks):                   LV-ViT-S (16 blocks):
┌─────────────────────────┐           ┌────────────────────────────────┐
│ Block 0-2: No compress  │           │ Block 0-3: No compress         │
│ Block 3: ★ COMPRESS     │           │ Block 4: ★ COMPRESS            │
│   τ₃ = 7.2×10⁻⁶        │           │   τ₄ = adaptive                │
│ Block 4-5: No compress  │           │ Block 5-7: No compress         │
│ Block 6: ★ COMPRESS     │           │ Block 8: ★ COMPRESS            │
│   τ₆ = 7.3×10⁻⁵        │           │   τ₈ = adaptive                │
│ Block 7-8: No compress  │           │ Block 9-11: No compress        │
│ Block 9: ★ COMPRESS     │           │ Block 12: ★ COMPRESS           │
│   τ₉ = 3.2×10⁻⁵        │           │   τ₁₂ = adaptive               │
│ Block 10-11: Head       │           │ Block 13-15: Head              │
└─────────────────────────┘           └────────────────────────────────┘
FLOPs: 4.6G → 2.9G (↓37%)           FLOPs: 6.6G → 4.6G (↓30%)
Top-1: 79.8% → 80.1%                 Top-1: 83.3% → 83.3%
```

---

## Visual Encoding Standard

| Symbol | Meaning |
|--------|---------|
| Solid grey patch | Pruned token (PPT: discarded) |
| **Striped grey patch** | Pruned token with squeezing (PPT++: info recovered) |
| Color-outlined patch | Pooled token (merged group) |
| **Spatially contiguous color group** | Spatial-aware BSM pool (PPT++) |
| Red star (★) | PPT++ contribution marker |
| Orange border | New/modified component |
| Yellow highlight | Adaptive threshold module |
| Green checkmark | Improvement over baseline |

---


This creates 5 publication-ready figures in `./figures/`:
1. `fig1_architecture.pdf` — PPT++ Architecture Overview
2. `fig2_variance_threshold.pdf` — Variance with Adaptive Threshold
3. `fig3_processing_steps.pdf` — Token Processing at Timesteps
4. `fig4_comparison.pdf` — PPT vs PPT++ Side-by-Side
5. `fig5_squeezing.pdf` — Token Squeezing Visualization