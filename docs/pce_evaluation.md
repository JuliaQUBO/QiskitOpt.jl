# Pauli Correlation Encoding (PCE) Evaluation (`QiskitOpt.PCE`)

This document evaluates the Pauli Correlation Encoding (PCE) solver from the
QSplit benchmark (`solver_pce.py`, following Sciorilli et al., *Towards
Large-Scale Quantum Optimization Solvers with Few Qubits*, 2024) for
`QiskitOpt.jl` ([`JuliaQUBO/QiskitOpt.jl#72`](https://github.com/JuliaQUBO/QiskitOpt.jl/issues/72)).

## 1. Algorithmic Overview and Qubit Compression

Instead of mapping $N$ binary decision variables $x_1, \dots, x_N \in \{0, 1\}$
one-to-one onto $N$ qubits in the computational basis, PCE compresses $N$
variables into $m \ll N$ encoding qubits by associating each variable $x_i$ with
a distinct weight-$k$ Pauli string observable $\Pi_i \in \{I, X, Y, Z\}^{\otimes m}$
 drawn from three mutually anticommuting single-type families:

$$\mathcal{P}_k^{(X)} = \{X_S : |S| = k\}, \quad \mathcal{P}_k^{(Y)} = \{Y_S : |S| = k\}, \quad \mathcal{P}_k^{(Z)} = \{Z_S : |S| = k\}.$$

Each family provides $\binom{m}{k}$ distinct observables, yielding a total
capacity of

$$N_{\max}(m, k) = 3 \binom{m}{k}$$

binary variables on $m$ qubits. For the standard two-body parity weight $k = 2$
used in QSplit's `solver_pce.py`, $N_{\max}(m, 2) = \frac{3}{2} m(m - 1)$, so the
encoding register scales as $m = O(\sqrt{N})$:

| Encoding qubits $m$ | observables per family $\binom{m}{2}$ | Max binary variables $N_{\max}(m, 2) = 3\binom{m}{2}$ |
| ---: | ---: | ---: |
| 2 | 1 | 3 |
| 3 | 3 | 9 |
| 4 | 6 | 18 |
| 5 | 10 | 30 |
| 6 | 15 | 45 |
| 8 | 28 | 84 |
| 10 | 45 | 135 |

## 2. Observable Count and `EstimatorV2` Call Cost

At each variational parameter vector $\theta$, PCE estimates the $N$ Pauli
expectations

$$e_i(\theta) = \langle \psi(\theta) | \Pi_i | \psi(\theta) \rangle \in [-1, 1], \qquad i = 1, \dots, N.$$

- **Measurement basis grouping**: Every observable in $\mathcal{P}_k^{(X)}$ is
  diagonal in the global $X^{\otimes m}$ basis, every observable in
  $\mathcal{P}_k^{(Y)}$ is diagonal in $Y^{\otimes m}$, and every observable in
  $\mathcal{P}_k^{(Z)}$ is diagonal in $Z^{\otimes m}$. Consequently, Qiskit's
  `EstimatorV2` can evaluate all $N$ observables in at most **3 Abelian
  measurement groups** per function evaluation regardless of $k$, whereas naive
  per-observable estimation would require $N$ circuits.
- **Classical surrogate evaluation**: Given the $N$-vector of expectations
  $e(\theta)$, computing the relaxed quadratic objective costs $O(N^2)$ floating
  point operations on the classical host per COBYLA step.
- **Ansatz expressibility vs barren-plateau / correlation bounds**: Because $N$
  variables share $m$ qubits, their expectation values $e_i(\theta)$ cannot vary
  independently across the entire $[-1, 1]^N$ hypercube; monogamy of
  entanglement and anticommutation relations bound the simultaneous magnitudes of
  $\langle \Pi_i \rangle$.

## 3. Smooth `tanh` Surrogate Loss vs True QUBO Energy

To make the objective differentiable/smooth for SciPy COBYLA, QSplit's
`solver_pce.py` maps each Pauli expectation $e_i(\theta)$ through a scaled
hyperbolic tangent:

$$\tilde{x}_i(\theta) = \frac{1 + \tanh(\alpha_{\text{pce}} \, e_i(\theta))}{2} \in (0, 1),$$

with sharpness hyperparameter $\alpha_{\text{pce}} > 0$ (typically $\alpha_{\text{pce}} \in [1.5, 4.0]$), and minimizes the smooth surrogate loss

$$\mathcal{L}_{\text{PCE}}(\theta) = \sum_{i=1}^N L_i \, \tilde{x}_i(\theta) + \sum_{i=1}^N \sum_{j=1}^N Q_{ij} \, \tilde{x}_i(\theta) \tilde{x}_j(\theta).$$

After COBYLA terminates at $\theta^*$, PCE threshold-decodes a discrete binary
assignment from the signs of the expectation values:

$$x_i^* = \begin{cases} 1, & e_i(\theta^*) \ge 0, \\ 0, & e_i(\theta^*) < 0. \end{cases}$$

**Critical scoring invariant**:
1. For fractional $\tilde{x}_i(\theta) \in (0, 1)$, $\tilde{x}_i(\theta)^2 \ne \tilde{x}_i(\theta)$, and $\mathcal{L}_{\text{PCE}}(\theta)$ omits the QUBO scale/offset transform $(\alpha, \beta)$ or includes regularization terms.
2. Therefore, $\mathcal{L}_{\text{PCE}}(\theta^*)$ is **never** reported as the `SampleSet` objective value.
3. Returned binary states $x^* \in \{0, 1\}^N$ are always scored against the
   **exact original QUBO** via `QUBOTools.value(x, L, Q, α, β)` (with sign
   flip for `MOI.MAX_SENSE`), while the final surrogate loss is recorded
   separately under `metadata["pce"]["surrogate_loss"]`.

## 4. Single-Solution Decoding vs Sample-Distribution Semantics

| Property | `QAOA.Optimizer` / `VQE.Optimizer` / `GAS.Optimizer` | `PCE.Optimizer` (Pauli Correlation Encoding) |
| --- | --- | --- |
| Qubit count for $N$ variables | $N$ qubits (plus value ancillas for `GAS`) | $m = O(\sqrt{N})$ qubits for weight-2 Pauli strings |
| Role of `NumberOfReads` | Shots in computational-basis sampling producing a multi-state `SampleSet` distribution | Shots used by `EstimatorV2` to estimate expectations $\langle \Pi_i \rangle_{\theta^*}$ |
| Decoded output | Empirical distribution over bitstrings $x \in \{0, 1\}^N$ | Single deterministic bitstring $x^* = \mathbb{I}[e(\theta^*) \ge 0] \in \{0, 1\}^N$ |

Because PCE decodes a single point estimate $x^*$ from expectation signs rather
than sampling a distribution of bitstrings from the quantum register, it should
not be conflated with computational-basis samplers in distribution-level
benchmarks.

## 5. Classical 1-Bit-Flip Polish Attribution

In QSplit's `solver_pce.py`, sign decoding is immediately followed by a greedy
classical 1-bit-flip local search (`polish`) that flips single variables whenever
the QUBO objective improves. On small and medium QUBOs, 1-flip local search
alone finds local minima rapidly, which can mask poor quantum expectation
decoding if enabled silently.

To maintain strict scientific attribution in `QiskitOpt.jl`:
- Classical bit-flip polish is **disabled by default** (`PCE.Polish() == false`).
- When explicitly opted into (`MOI.set(model, PCE.Polish(), true)`),
  `metadata["pce"]` records both the unpolished quantum-decoded assignment and
  its true QUBO energy (`raw_quantum_state`, `raw_quantum_value`) alongside the
  post-polish assignment, energy, flip count, and pass count (`polish_enabled`,
  `polish_flips`, `polish_passes`, `polished_value`).

## 6. Go / No-Go Architectural Decision

- **Decision**: Ship `QiskitOpt.PCE` as an **experimental submodule** (`src/PCE.jl`, exported as `QiskitOpt.PCE`) rather than promoting it to a primary default computational-basis sampler alongside `QAOA` and `VQE`.
- **Rationale**:
  1. `QiskitOpt.PCE` gives users and benchmark workflows a credential-free, self-contained implementation of QSplit's Pauli correlation encoding (`PCE.minimum_pce_qubits`, `PCE.pauli_correlation_observables`, `PCE.surrogate_variables`, `PCE.surrogate_loss`, `PCE.decode_expectations`, `PCE.bitflip_polish`, and `PCE.Optimizer`) without adding any new Python dependencies.
  2. Labeling `PCE` as experimental and keeping `PCE.Polish()` disabled by default preserves a clean boundary between multi-sample quantum circuit distributions (`QAOA`, `VQE`, `GAS`) and expectation-sign decoding with optional classical local search.

