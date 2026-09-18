# MCTR v3 — Complete Scalable Architecture, Mathematics, and Implementation

## 1. Purpose

MCTR (Meta-Cognitive Transformer Regulation) is a causal closed-loop Transformer architecture:

\[
\boxed{\text{Inference}\rightarrow\text{Observation}\rightarrow\text{Meta-Evaluation}\rightarrow\text{Regulation}\rightarrow\text{Refined Inference}\rightarrow\text{Re-Evaluation}}
\]

Meta-cognition is operationalized as second-order inference over measurable computational state. No claim of consciousness or subjective awareness is required.

## 2. Core v3 Architecture

The core state is:

\[
\Psi_t=[C_t,U_t,N_t,K_t,S_t,\hat H_t]
\]

The meta-evaluator is:

\[
m_t=M_\phi(\Psi_t)
\]

The controller is:

\[
\Gamma_{t+1}=G_\omega(m_t)
\]

The initial control vector is:

\[
\Gamma_t=[T_t,\alpha_t]
\]

where \(T_t\) regulates output decisiveness and \(\alpha_t\) regulates meta-attention strength.

The canonical recurrence is:

\[
\boxed{
\Psi_t=
\mathcal E\left[
F_\theta\left(
X_t;G_\omega(M_\phi(\Psi_{t-1}))
\right)
\right]
}
\]

This is the central mathematical definition of MCTR v3.

## 3. First-Order Transformer

At step \(t\):

\[
h_t=F_\theta(X_t;\Gamma_t)
\]

\[
z_t=W_oh_t+b_o
\]

\[
P_t(i)=\frac{\exp(z_{t,i})}{\sum_j\exp(z_{t,j})}
\]

Temperature-controlled output:

\[
\boxed{
P_t(i;T_t)=
\frac{\exp(z_{t,i}/T_t)}
{\sum_j\exp(z_{t,j}/T_t)}
}
\]

Lower \(T_t\) produces a sharper distribution; higher \(T_t\) produces a flatter distribution.

## 4. Information-Theoretic State

Predictive entropy:

\[
\boxed{
H_t=-\sum_iP_t(i)\log P_t(i)
}
\]

Normalized entropy:

\[
\boxed{
\hat H_t=\frac{H_t}{\log|V|}
}
\]

Hence:

\[
0\leq\hat H_t\leq1.
\]

For fixed non-degenerate logits:

\[
\frac{\partial H_t}{\partial T_t}>0.
\]

The important distinction is:

\[
T_t\rightarrow H_t
\]

is the statistical relationship of softmax temperature to entropy, while:

\[
\Psi_t\rightarrow T_t
\]

is the MCTR control relationship. No linear \(T\propto H\) assumption is required.

## 5. Meta-Cognitive State Variables

\[
\boxed{\Psi_t=[C_t,U_t,N_t,K_t,S_t,\hat H_t]}
\]

### Confidence

\[
C_t=\max_iP_t(i)
\]

or:

\[
C_t=P_t^{[1]}-P_t^{[2]}.
\]

### Uncertainty

Baseline:

\[
U_t=\hat H_t
\]

Learned alternative:

\[
U_t=f_u(h_t,P_t,H_t).
\]

### Novelty

\[
N_t=D(h_t,\mathcal D_{\mathrm{ref}})
\]

with possible implementation:

\[
N_t=1-\cos(h_t,\mu_{\mathrm{ref}})
\]

or a validated Mahalanobis score.

### Conflict

Use genuinely distinct computational paths:

\[
K_t=D_{JS}(P_t^{[a]},P_t^{[b]})
\]

where:

\[
D_{JS}(P,Q)=\frac12D_{KL}(P\|M)+\frac12D_{KL}(Q\|M)
\]

and:

\[
M=\frac12(P+Q).
\]

Do not confuse this with the top-two probability margin.

### Stability

\[
\Delta h_t=h_t-h_{t-1}
\]

\[
\Delta P_t=P_t-P_{t-1}
\]

and a candidate estimator is:

\[
S_t=
\exp[-\lambda_h\|\Delta h_t\|_2-\lambda_pD(P_t,P_{t-1})].
\]

These estimators are hypotheses and must be experimentally validated.

## 6. Meta-Cognitive Evaluator

\[
\boxed{m_t=M_\phi(\Psi_t)}
\]

A minimal implementation:

\[
m_t=W_2\sigma(W_1\Psi_t+b_1)+b_2.
\]

Conceptually:

\[
\Psi_t:\text{What is happening in my inference?}
\]

\[
m_t:\text{How should computation respond?}
\]

## 7. Adaptive Entropy Target

Define:

\[
H_t^*=f_H(\Psi_t)
\]

and:

\[
e_t=H_t^*-H_t.
\]

Then:

\[
\boxed{
T_{t+1}=
\operatorname{clip}
(T_t+\eta_Te_t,T_{\min},T_{\max})
}
\]

The control interpretation is:

\[
H_t=\text{observed state}
\]

\[
H_t^*=\text{desired state}
\]

\[
e_t=\text{control error}
\]

\[
T_t=\text{control action}.
\]

## 8. Why Attention Is Separate From Temperature

Temperature controls:

\[
\boxed{\text{How decisively should the model choose?}}
\]

Meta-attention controls:

\[
\boxed{\text{What information should receive computational priority?}}
\]

Therefore v3 does not make attention temperature a required second entropy loop.

Initially:

\[
\boxed{T_t^A=1}
\]

Adaptive \(T_t^A\) is an optional later extension.

## 9. Meta-Cognitive Attention

Project the previous meta-state:

\[
M_t=W_mm_{t-1}.
\]

Then:

\[
\boxed{
A_t^{MC}
=
\operatorname{Softmax}
\left(
\frac{Q_tK_t^T}{\sqrt{d_k}}
+\alpha_tM_t
\right)
}
\]

and:

\[
\boxed{
h_t^{MC}=A_t^{MC}V_t.
}
\]

This allows the meta-state to alter information allocation without introducing an additional adaptive temperature.

## 10. Refined Inference

\[
z_t^{MC}=W_oh_t^{MC}+b_o
\]

\[
\boxed{
P_t^{MC}(i)=
\frac{\exp(z_{t,i}^{MC}/T_t)}
{\sum_j\exp(z_{t,j}^{MC}/T_t)}
}
\]

Then:

\[
y_t\sim P_t^{MC}
\]

or:

\[
y_t=\arg\max_iP_t^{MC}(i).
\]

## 11. Post-Regulation Re-Evaluation

After regulation:

\[
H_t^{MC}=-\sum_iP_t^{MC}(i)\log P_t^{MC}(i).
\]

Then recompute:

\[
\boxed{
\Psi_{t+1}
=
\mathcal E(h_t^{MC},P_t^{MC},\text{history})
}
\]

The intervention is therefore evaluated rather than assumed successful.

## 12. Causal Computational Graph

Invalid circular formulation:

\[
P_t\rightarrow\Psi_t\rightarrow\text{modify the attention that produced }P_t.
\]

Recommended causal ordering:

\[
\boxed{
\Psi_{t-1}
\rightarrow
m_{t-1}
\rightarrow
\Gamma_t
\rightarrow
\text{Inference}_t
\rightarrow
\Psi_t
}
\]

Thus the previous inferential state controls the current computation.

## 13. Full Controller

\[
\boxed{
\Gamma_t=G_\omega(m_{t-1})
}
\]

Initial:

\[
\Gamma_t=[T_t,\alpha_t].
\]

Optional later:

\[
\Gamma_t=[T_t,\alpha_t,\beta_t,d_t].
\]

Keep \(\beta_t\) and \(d_t\) disabled until the core mechanism is validated.

## 14. Full Causal Recurrence

\[
\boxed{
\Psi_{t-1}\rightarrow m_{t-1}\rightarrow\Gamma_t\rightarrow h_t\rightarrow P_t\rightarrow\Psi_t
}
\]

This has the form of a controlled dynamical system:

\[
\text{State}_{t-1}\rightarrow\text{Controller}\rightarrow\text{Action}_t\rightarrow\text{State}_t.
\]

## 15. Canonical MCTR Equation

\[
\boxed{
\Psi_t=
\mathcal E
\left[
F_\theta
\left(
X_t;
G_\omega(M_\phi(\Psi_{t-1}))
\right)
\right]
}
\]

This means the model's previous inferential state determines how its next computation is performed.

## 16. Training Objective

\[
\boxed{
\mathcal L=
\mathcal L_{\mathrm{task}}
+\lambda_C\mathcal L_{\mathrm{cal}}
+\lambda_U\mathcal L_U
+\lambda_R\mathcal L_R
+\lambda_S\mathcal L_S
}
\]

Task:

\[
\mathcal L_{\mathrm{task}}
=
-\sum_t\log P_t^{MC}(y_t^*).
\]

Calibration:

\[
\mathcal L_{\mathrm{cal}}=(C_t-Y_t)^2.
\]

Uncertainty:

\[
\mathcal L_U=
-[(1-Y_t)\log U_t+Y_t\log(1-U_t)].
\]

Regulation smoothness:

\[
\mathcal L_R=
\sum_t\|\Gamma_t-\Gamma_{t-1}\|_2^2.
\]

Meta-state smoothness:

\[
\mathcal L_S=
\sum_t\|m_t-m_{t-1}\|_2^2.
\]

Do not add arbitrary entropy regularization simply to force entropy toward a preferred value.

## 17. Degenerate-Regulation Guard

A constant controller can minimize:

\[
\mathcal L_R.
\]

Therefore:

\[
\boxed{
\text{low regulation loss}\neq\text{good regulation}.
}
\]

Evaluate both stability and responsiveness:

\[
\rho_{TH}=\operatorname{Corr}(T_t,H_t).
\]

A useful controller should be stable and responsive.

## 18. Gradient and Training Flow

Use teacher forcing for the first implementation.

Avoid initial training through discrete self-generated tokens.

Later alternatives include:

- policy gradients,
- Gumbel-Softmax,
- differentiable soft refinement.

These are optional extensions, not requirements for the first proof of concept.

## 19. Premise Validation

Before training the controller, freeze the base Transformer.

Collect:

\[
\{(\hat H_t,C_t,Y_t)\}_{t=1}^{N}.
\]

Evaluate:

\[
AUROC(\hat H_t,\mathrm{error})
\]

\[
AUROC(C_t,\mathrm{error})
\]

plus:

- ECE,
- Brier score,
- reliability diagrams.

If entropy/confidence are weak error predictors, investigate the premise before adding architectural complexity.

## 20. Runtime Algorithm

For each generation step:

1. Retrieve previous state \(\Psi_{t-1}\).
2. Compute \(m_{t-1}=M_\phi(\Psi_{t-1})\).
3. Compute \(\Gamma_t=G_\omega(m_{t-1})\).
4. Extract \(T_t,\alpha_t\).
5. Compute \(Q_t,K_t,V_t\).
6. Compute \(M_t=W_mm_{t-1}\).
7. Compute:
   \[
   A_t^{MC}=\operatorname{Softmax}(Q_tK_t^T/\sqrt{d_k}+\alpha_tM_t).
   \]
8. Compute:
   \[
   h_t^{MC}=A_t^{MC}V_t.
   \]
9. Compute logits.
10. Apply \(T_t\).
11. Generate \(y_t\).
12. Compute entropy and all state estimators.
13. Form \(\Psi_t\).
14. Append \(y_t\) to the context.
15. Repeat.

Pseudocode:

```text
initialize Ψ
initialize T

for t in generation:

    m = MetaEvaluator(Ψ)
    Γ = Controller(m)

    T = bounded_temperature(Γ)
    α = attention_strength(Γ)

    Q, K, V = TransformerProjections(X_t)
    M = W_m m

    A_MC = softmax(QKᵀ / sqrt(d_k) + αM)
    h_MC = A_MC V

    logits = W_o h_MC + b_o
    P = softmax(logits / T)

    y_t = sample_or_argmax(P)

    H = entropy(P)
    C = confidence(P)
    U = uncertainty(h_MC, P, H)
    N = novelty(h_MC)
    K_conflict = conflict_estimator(...)
    S = stability(h_MC, P, history)

    Ψ = [C, U, N, K_conflict, S, normalized(H)]

    append y_t to X

return generated sequence
```

## 21. Scalable Software Architecture

```text
mctr/
├── config/
│   ├── model.yaml
│   ├── controller.yaml
│   └── experiments.yaml
├── model/
│   ├── base_transformer.py
│   ├── mc_attention.py
│   ├── temperature_controller.py
│   └── mctr_model.py
├── state/
│   ├── entropy.py
│   ├── confidence.py
│   ├── uncertainty.py
│   ├── novelty.py
│   ├── conflict.py
│   └── stability.py
├── controller/
│   ├── meta_evaluator.py
│   └── controller.py
├── training/
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── calibration.py
│   ├── uncertainty.py
│   ├── attention.py
│   ├── temperature.py
│   └── efficiency.py
└── logging/
    └── trajectory_logger.py
```

Every state estimator should be replaceable independently.

## 22. Scalability Path

### Level 1

\[
\Psi_t\rightarrow T_t
\]

Adaptive temperature only.

### Level 2

\[
\Psi_t\rightarrow M_t,\alpha_t
\]

Meta-attention.

### Level 3

\[
\Psi_t\rightarrow[T_t,M_t,\alpha_t]
\]

Full MCTR v3.

### Level 4

Add:

\[
T_t^A,\beta_t,d_t.
\]

Do not advance levels until the preceding level passes ablation.

## 23. Mathematical Verification

Check:

\[
\sum_iP_t(i)=1
\]

\[
0\le H_t\le\log|V|
\]

\[
0\le\hat H_t\le1
\]

\[
T_{\min}\le T_t\le T_{\max}
\]

\[
\sum_jA_{t,j}^{MC}=1
\]

\[
0\le K_t\le\log2
\]

and, for the exponential stability estimator:

\[
0<S_t\le1.
\]

Causality requirement:

\[
\boxed{
\Gamma_t=f(\Psi_{t-1})
}
\]

for the recommended one-step implementation.

## 24. Numerical Stability

Use log-softmax for probability computation.

Compute entropy from log-probabilities.

Keep:

\[
T_t\ge\epsilon.
\]

Use stable Jensen-Shannon/KL implementations.

For Mahalanobis novelty, use linear solves/Cholesky rather than explicit matrix inversion.

## 25. Evaluation Ladder

### Phase −1

Base-model premise validation.

### Phase 0

Base benchmark.

### Phase 1

Adaptive temperature.

### Phase 2

Meta-attention.

### Phase 3

Full MCTR.

### Phase 4

Optional adaptive attention temperature and deeper controls.

Recommended first benchmark:

\[
\boxed{\text{GSM8K}}
\]

Then:

- MMLU,
- ARC-Challenge,
- BBH/BBEH,
- controlled OOD,
- contradiction/conflict data,
- repeated-seed stability,
- dedicated MCTR-Eval.

## 26. Ablation Matrix

| Model | Meta-State | Adaptive T | MC-Attention | Adaptive T^A |
|---|---:|---:|---:|---:|
| Base | No | No | No | No |
| Adaptive-T | Yes | Yes | No | No |
| MC-Attention | Yes | No | Yes | No |
| Full MCTR | Yes | Yes | Yes | No |
| MCTR + T^A | Yes | Yes | Yes | Yes |

Also test removal of entropy, confidence, uncertainty, novelty, conflict, and stability individually.

## 27. Evaluation Metrics

Capability:

\[
Accuracy,\quad Loss.
\]

Calibration:

\[
ECE,\quad Brier.
\]

Error awareness:

\[
AUROC(U,\mathrm{error}),\quad AUPRC(U,\mathrm{error}).
\]

Novelty:

\[
AUROC(N,\mathrm{OOD}).
\]

Conflict:

\[
AUROC(K,\mathrm{conflict}).
\]

Self-correction:

\[
\boxed{
\Delta Accuracy=
Accuracy_{\mathrm{refined}}-Accuracy_{\mathrm{initial}}
}
\]

Temperature:

\[
Corr(T,H),\quad |H-H^*|,\quad Var(T),\quad \sum_t(T_t-T_{t-1})^2.
\]

Efficiency:

- latency,
- throughput,
- peak memory,
- additional compute.

## 28. Attention Evaluation

Attention entropy:

\[
H_t^A=-\sum_jA_{t,j}^{MC}\log A_{t,j}^{MC}.
\]

Attention concentration:

\[
C_t^A=\max_jA_{t,j}^{MC}.
\]

Redistribution:

\[
D_t^A=D_{JS}(A_t^{Base},A_t^{MC}).
\]

Attention changes must be validated causally using masking, perturbation, intervention, or output-sensitivity tests. Visual attention differences alone are insufficient.

## 29. Optional Adaptive Attention Temperature

Only after core validation:

\[
H_t^A=-\sum_jA_{t,j}^{MC}\log A_{t,j}^{MC}
\]

\[
H_t^{A*}=f_A(\Psi_t)
\]

\[
e_t^A=H_t^{A*}-H_t^A
\]

\[
T_{t+1}^A=
\operatorname{clip}
(T_t^A+\eta_Ae_t^A,T_{\min}^A,T_{\max}^A).
\]

This is an extension, not a prerequisite.

## 30. Computational Cost

The meta-controller adds approximately:

\[
O(d_md_\Psi+d_\Gamma d_m).
\]

Standard attention remains approximately:

\[
O(n^2d).
\]

The meta-attention bias adds approximately:

\[
O(n^2)
\]

depending on implementation.

Independent-path conflict can add forward/decoding cost.

Two-pass alternatives can approach:

\[
Cost\approx2\times.
\]

Report latency, memory, throughput, and extra forward passes.

## 31. Final Mathematical System

\[
\boxed{
\begin{aligned}
\Psi_t
&=\mathcal E(h_t,P_t,\mathcal H_t)\\
m_t
&=M_\phi(\Psi_t)\\
\Gamma_{t+1}
&=G_\omega(m_t)\\
M_{t+1}
&=W_mm_t\\
A_{t+1}^{MC}
&=\operatorname{Softmax}
\left(
Q_{t+1}K_{t+1}^{T}/\sqrt{d_k}
+\alpha_{t+1}M_{t+1}
\right)\\
h_{t+1}^{MC}
&=A_{t+1}^{MC}V_{t+1}\\
P_{t+1}^{MC}
&=\operatorname{Softmax}
\left(
W_oh_{t+1}^{MC}/T_{t+1}
\right)\\
\Psi_{t+1}
&=\mathcal E(h_{t+1}^{MC},P_{t+1}^{MC},\mathcal H_{t+1}).
\end{aligned}
}
\]

## 32. Central Research Proposition

The system tests:

\[
\boxed{
\textit{Can a measurable estimate of a model's current computational state improve regulation of its subsequent inference?}
}
\]

The complete evidence chain is:

\[
\boxed{
\text{State signal}
\rightarrow
\text{Error prediction}
\rightarrow
\text{Meta-evaluation}
\rightarrow
\text{Regulation}
\rightarrow
\text{Improved state}
\rightarrow
\text{Improved reliability}
}
\]

This is the scalable mathematical and implementation definition of MCTR v3.
