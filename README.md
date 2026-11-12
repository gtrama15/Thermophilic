# Thermophilic

Analysis code for residue-pair-resolved amide-I vibrational exciton coupling in the
phosphorylated (1QMP) and unphosphorylated (1DZ3) forms of the *Geobacillus
stearothermophilus* Spo0A receiver domain. Code accompanying the manuscript submitted
to *Phys. Chem. Chem. Phys.*

## Repository layout
- `physics/`: electrostatic frequency map, TDC coupling, exciton Hamiltonian, and
  2D-IR response calculation (`calculate_2dir_response.py`)
- `analysis/`: ensemble definition, decomposition, robustness checks, and summary tables
- `figures/`: scripts that regenerate the manuscript figures and tables, including the
  validation benchmarks of the 2D-IR code

See the manuscript Methods section for the full description of each step.

## 2D-IR response code
`physics/calculate_2dir_response.py` computes the third-order response (rephasing and
non-rephasing; GSB, SE and ESA pathways) for the two-exciton amide-I model with ZZZZ
orientational averaging and static ensemble averaging. Defaults: Γ = 4 cm⁻¹
(homogeneous HWHM), Δ = 16 cm⁻¹ (local-mode anharmonicity).

**Note (Nov 2026):** the rephasing ω₁ axis is now mirrored (`lorentzian(-W1, ...)`) in
the GSB, SE and ESA terms, so rephasing and non-rephasing resonances coincide. Earlier
versions did not do this, and spectra from them should not be used. The TDC and
frequency-map analysis (Table 1, distance and orientation decomposition) does not use
this code and is unaffected.

## Validation
- Single oscillator vs analytic closed form: max relative error 2.9 × 10⁻¹⁰.
- Coupled dimer (ωA = 1650, ωB = 1630, J = 10 cm⁻¹, Δ = 16, Γ = 4 cm⁻¹) vs an
  independent 9-level exact diagonalization: correlation 1.000000.
- Scripts: `figures/make_figureS6_validation.py` (Figure S6) and
  `figures/benchmark_coupled_dimer_cross_check.py`. Run them from inside `figures/`;
  they import from `../physics`.

## Data availability
Molecular dynamics trajectories and intermediate per-snapshot coupling data are
available from the corresponding author upon reasonable request.
