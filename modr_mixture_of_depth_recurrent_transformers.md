# MoDr: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning

**Published as a conference paper at ICLR 2026**

**Authors:**
Xiaojing Zhang¹, Haifeng Wu³, Gang He¹, Jiyang Shen¹, Bochen Lyu²˒¹*, Zhanxing Zhu²*
* ¹DataCanvas
* ²University of Southampton
* ³Find AI
* \*Corresponding authors: `bochen.lyu@soton.ac.uk`, `zhanxing.zhu@soton.ac.uk`

**Code Repository:** [https://github.com/zhangxjohn/MoDr](https://github.com/zhangxjohn/MoDr)

---

## ABSTRACT

Large Language Models have demonstrated superior reasoning capabilities by generating step-by-step reasoning in natural language before deriving the final answer. However, this process often verbalizes excessively long intermediate reasoning (referred to as “deep thinking”) before delivering an answer, which significantly increases test-time computational costs. To mitigate this issue, recent studies have explored latent reasoning, which allows models to conduct continuous “deep thinking” in the continuous latent space without token generation, achieving promising performance while reducing computational overhead. In this paper, we focus on the depth-recurrent Transformers (such as Huginn), which repeats the computation across the depth of standard Transformer layers, and identify that their single sequential recurrent mechanism constrains the search space for latent reasoning.

To overcome this limitation, we introduce **Mixture-of-Depth-Recurrent (MoDr) Transformers**, a novel dynamic routing framework designed to broaden the reasoning search space in depth-recurrent models. Specifically, we equip the recurrent module with multiple candidate branches instantiated via Low-Rank Adaptation (LoRA) and design a hard-gate router to dynamically select the optimal branch for each token during inference based on the current context. Furthermore, we develop an auxiliary-loss-free load balancing strategy to ensure balanced training across all branches. Extensive experiments on mathematical reasoning, commonsense reasoning, and code generation benchmarks demonstrate that our proposed MoDr significantly improves upon the performance of the existing Huginn model (e.g., **+7.20%** on mathematical reasoning tasks and **+21.21%** on commonsense tasks) with negligible additional computational overhead.

---

## 1 INTRODUCTION

Large Language Models (LLMs) (Achiam et al., 2023; Touvron et al., 2023; DeepSeek-AI et al., 2025) have achieved remarkable success across diverse tasks, particularly in complex multi-step reasoning. To tackle complex problems, recent approaches typically elicit chain-of-thought (CoT) reasoning (Wei et al., 2022b; Kojima et al., 2022) in natural language before generating the final answer. However, natural language is inherently discrete, non-differentiable, and prone to error accumulation, often requiring models to generate excessively long token sequences to complete reasoning, which incurs substantial test-time computation and memory overhead.

To address these challenges, recent research has shifted focus toward **latent reasoning** (Saunshi et al., 2025; Geiping et al., 2025; Huang et al., 2025; Labovich, 2025; Goyal et al., 2021), which conducts continuous computation in the hidden activation space rather than verbalizing tokens. Among these, **depth-recurrent Transformers** (such as Huginn (Geiping et al., 2025) and Universal Transformer (Dehghani et al., 2019)) have emerged as a prominent paradigm. These models reuse a core stack of Transformer layers iteratively, achieving depth adaptivity and recurrent thinking.

However, existing depth-recurrent architectures suffer from a critical limitation: their recurrent core follows a **single, rigid sequential forward path**. From a cognitive psychology perspective, problem solving involves two complementary processes:
1. **Exploration:** Searching through diverse problem formulations, alternative reasoning paths, and perspectives.
2. **Rumination (Exploitation):** Deliberating deeply and iteratively refining candidate intermediate thoughts.

Vanilla depth-recurrent models (like Huginn) excel at rumination via tied layer loops, but their single-path design severely constrains exploration. To break free from this restriction, we propose **Mixture-of-Depth-Recurrent (MoDr) Transformers**.

### Key Contributions:
1. **Dynamic Multi-Branch Recurrent Architecture:** We augment the depth-recurrent loop with multiple candidate reasoning branches parameterized by LoRA modules (sharing the underlying Transformer block weights), enabling diverse reasoning trajectories without memory explosion.
2. **Hard-Gate Dynamic Routing:** We introduce a lightweight router that dynamically selects the most appropriate recurrent branch for each token at each reasoning step based on prelude embeddings and recurrent hidden states.
3. **Auxiliary-Loss-Free Load Balancing:** We develop a sequence-level load-balancing mechanism with dynamic branch bias adjustment $\Delta b_i$, preventing routing collapse without introducing conflicting auxiliary loss gradients.
4. **Strong Empirical Performance:** MoDr achieves substantial gains over Huginn and fine-tuned baselines across math, commonsense, and coding benchmarks while preserving inference speed and deployment memory footprint.

---

> **Figure 1: Comparison of (a) vanilla Huginn model and (b) MoDr (ours)**
> - **(a) Vanilla Depth-Recurrent Huginn Model:** $x \to \text{Prelude } \mathbf{P} \to \text{Recurrent Core } \mathbf{R} \text{ (Loop for } T \text{ steps with state } \mathbf{s}_t \text{ and input injection } \mathbf{e}) \to \text{Coda } \mathbf{C} \to \text{Answer}$. All tokens pass sequentially through the same tied recurrent blocks.
> - **(b) Mixture-of-Depth-Recurrent Transformer (MoDr, Ours):** $x \to \text{Prelude } \mathbf{P} \to \text{Multi-Branch Recurrent Module } \{\mathbf{R}_1, \mathbf{R}_2, \dots, \mathbf{R}_N\} \text{ governed by Hard-Gate Router } \to \text{Coda } \mathbf{C} \to \text{Answer}$. Dynamic token-level routing takes turns across specialized LoRA branches during reasoning steps.

---

## 2 BACKGROUND

### 2.1 Depth Adaptivity in Transformers
To scale test-time computation while reducing parameter footprint, looped and depth-recurrent Transformers execute a set of layers multiple times. Formally, given an input token sequence $\mathbf{x} = [x_1, x_2, \dots, x_n]$, where $x_i \in \mathbb{R}^{|\mathcal{V}|}$, $n$ denotes the length of the input, and $|\mathcal{V}|$ is the vocabulary size, AlgoFormer (Gao et al., 2024b) and Depth-Recurrent Huginn (Geiping et al., 2025) proposed a three-stage Prelude/Loop/Coda structure formulated as:

$$ f = f_{\text{head}} \circ f_{\text{coda}} \circ \underbrace{f_R \circ \cdots \circ f_r \circ \cdots \circ f_1}_{T \text{ iterations}} \circ f_{\text{prelude}} \circ f_{\text{embed}} \tag{1} $$

The Huginn model architecture is structured around three functional modules: (1) 2 prelude blocks, which embed input context into latent space; (2) 4 recurrent blocks, which sequentially process the output from the prelude module; (3) 2 coda blocks, which decode from latent space to predict the next token:

$$ \mathbf{e} = \mathbf{P}(\mathbf{x}) \tag{2} $$
$$ \mathbf{s}_0 \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_{n \cdot h}) \tag{3} $$
$$ \mathbf{s}_t = \mathbf{R}(\mathbf{e}, \mathbf{s}_{t-1}) \quad \text{for} \quad t \in \{1, 2, \dots, T\} \tag{4} $$
$$ \mathbf{p} = \mathbf{C}(\mathbf{s}_T) \tag{5} $$

where $\mathbf{s}_0$ is a random Gaussian vector serving as the initial state of the recurrent module, $\sigma$ is a noise scaling factor, $T$ denotes the maximum number of recurrent iterations, and $\mathbf{p} \in \mathbb{R}^{n \times |\mathcal{V}|}$ is the output token distribution.

In Huginn, input injection is performed at each recurrent iteration: $\mathbf{z}_0 = \mathbf{s}_{t-1} + \mathbf{e}$. Each recurrent iteration applies $L$ standard Transformer sandwich layers. For each recurrent step $t \in \{1, 2, \dots, T\}$, the hidden state $\mathbf{z}_t^l$ is computed as:

$$ \hat{\mathbf{z}}_t^l = \text{LN}(\text{Attn}(\text{LN}(\mathbf{z}_t^{l-1}) \mid \mathbf{W}^l) + \mathbf{z}_t^{l-1}) \tag{6} $$
$$ \mathbf{z}_t^l = \text{LN}(\text{MLP}(\text{LN}(\hat{\mathbf{z}}_t^l) \mid \mathbf{W}^l) + \hat{\mathbf{z}}_t^l) \tag{7} $$

where $\mathbf{W}^l$ denotes the parameters of the $l$-th recurrent block, and $\mathbf{s}_t = \mathbf{z}_t^L$.

### 2.2 Exploration and Rumination in Problem Solving
Cognitive problem solving requires both broad exploratory hypothesis generation and focused ruminative refinement. While the recurrent loop provides rumination, a single tied block forces all tokens through identical transformations. MoDr addresses this by introducing branching exploration directly inside the recurrent depth loop.

---

> **Figure 2: The architecture of the Mixture-of-Depth-Recurrent (MoDr) Transformer.**
> - **(a) Overall Pipeline:** Prelude $\mathbf{P} \to$ Multi-Branch Recurrent Module $\{\mathbf{R}_1, \dots, \mathbf{R}_N\} \to$ Coda $\mathbf{C}$.
> - **(b) Hard-Gate Router:** Dynamic token selection based on context vector $\mathbf{h}$.
> - **(c) Multi-Branch Recurrent Module:** Shared frozen base Transformer blocks $\mathbf{W}_0$ augmented with branch-specific LoRA adapters $\{\Delta\mathbf{W}_j\}_{j=1}^N$.
> - **(d) Loss-Free Branch Bias Updating:** Dynamic bias shifts $+0.2, -0.1, -0.2, +0.0, +0.1$ adjusted per batch based on load violation error.

---

## 3 METHODOLOGY

### 3.1 Multi-Branch Recurrent Module
To enable exploration without multiplying parameter count or GPU memory, MoDr parameterizes $N$ distinct recurrent branches using Low-Rank Adaptation (LoRA) on top of shared base recurrent blocks $\mathbf{W}^l$:

$$ \hat{\mathbf{z}}_{j,t}^l = \text{LN}(\text{Attn}(\text{LN}(\mathbf{z}_{j,t}^{l-1}) \mid \mathbf{W}^l, \Delta\mathbf{W}_j^l) + \mathbf{z}_{j,t}^{l-1}) \tag{8} $$
$$ \mathbf{z}_{j,t}^l = \text{LN}(\text{MLP}(\text{LN}(\hat{\mathbf{z}}_{j,t}^l) \mid \mathbf{W}^l, \Delta\mathbf{W}_j^l) + \hat{\mathbf{z}}_{j,t}^l) \tag{9} $$

where $\{\Delta\mathbf{W}_j^l\}_{j=1}^N$ denotes the trained LoRA module parameters for $N$ recurrent branches. Specifically, for a base feature transformation $\mathbf{z} = \mathbf{W}_0 \mathbf{x}$, our modified forward pass yields:

$$ \mathbf{z} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \Delta\mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x} \tag{10} $$

where $\mathbf{B} \in \mathbb{R}^{h \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(h, k)$ for $h$ and $k$ being the dimensions of the original parameter matrix $\mathbf{W}_0$. The scaling factor $\alpha$ controls adaptation magnitude. The backbone model weights $\mathbf{W}_0$ remain frozen.

### 3.2 Hard-Gate Branch Routing Strategy
Inspired by sparsely-gated Mixture-of-Experts (MoE) (Shazeer et al., 2017) and Switch Transformer (Fedus et al., 2022), we design a learnable hard-gate routing network to determine which candidate recurrent branch will predict the next token according to the hidden state information $\mathbf{h} \in \mathbb{R}^{n \times h}$ derived from: (1) prelude output $\mathbf{e}$, and (2) recurrent state $\mathbf{s}$, mapped via adapter $\mathbb{R}^{2h} \to \mathbb{R}^h$.

Let $\mathbf{W}_{\text{router}} \in \mathbb{R}^{N \times h}$ denote the trainable weight matrix of the routing network. For Top-1 hard-gate routing:

$$ \mathbf{u} = \mathbf{W}_{\text{router}} \mathbf{h}^\top, \quad \mathbf{u} \in \mathbb{R}^{N \times n} \tag{11} $$
$$ \boldsymbol{r} = \sigma\left(\frac{1}{n} \sum_{i=1}^n (\mathbf{u}_i)\right), \quad \boldsymbol{r} \in \mathbb{R}^N \tag{12} $$
$$ \zeta = \arg\max_j (r_j), \quad j \in \{1, 2, \dots, N\} \tag{13} $$
$$ g = r_j \quad \text{if} \quad j = \zeta \tag{14} $$

where $\zeta$ is the index of the selected branch, $g$ is a scalar score, and $\sigma$ is a nonlinear activation function (e.g. sigmoid or softmax). The hidden states of the selected branch are modulated as:

$$ \mathbf{z}_{j,t}^l \leftarrow g \cdot \mathbf{z}_{j,t}^l $$

During inference, the hard-gate router dynamically selects which recurrent branch performs "deep thinking" token-by-token in a "relay race" fashion.

### 3.3 Auxiliary-Loss-Free Load Balancing
To prevent routing collapse (Shazeer et al., 2017) without introducing conflicting auxiliary loss gradients (Wang et al., 2024), we add a bias term $\{b_i\}_{i=1}^N$ to the gating score:

$$ \hat{\boldsymbol{r}} = \boldsymbol{r} + \boldsymbol{b}, \quad \hat{\boldsymbol{r}} \in \mathbb{R}^N \tag{15} $$
$$ \hat{\zeta} = \arg\max_j (\hat{r}_j), \quad j \in \{1, 2, \dots, N\} \tag{16} $$
$$ g = r_j \quad \text{if} \quad j = \hat{\zeta} \tag{17} $$

*(Note: the multiplication weight $g$ uses the unbiased confidence score $r_{\hat{\zeta}}$.)*

To adjust $b_i$ during training, each $b_i$ is initialized to 0. For each batch, the assigned count $c_i$ and mean count $\bar{c}_i$ are tracked:

$$ e_i = \bar{c}_i - c_i \tag{18} $$
$$ b_i \leftarrow b_i + \eta * \text{sign}(e_i) \tag{19} $$

where $\eta$ is the bias update rate (e.g. $\eta = 0.001$), and $\text{sign}(\cdot)$ is the sign function.

---

## 4 EXPERIMENTS

### 4.1 Experimental Setup
- **Tasks & Datasets:** Six mathematical reasoning tasks:
  1. **GSM8K** (Cobbe et al., 2021) — Grade school math word problems (In-Domain).
  2. **MAWPS** (Koncel-Kedziorski et al., 2016) — Arithmetic & algebra repository (In-Domain).
  3. **AQuA** (Ling et al., 2017) — Algebraic word problems (In-Domain).
  4. **MultiArith** (Roy & Roth, 2016) — Multi-step arithmetic word problems (Out-of-Domain).
  5. **AddSub** (Hosseini et al., 2014) — Addition & subtraction word problems (Out-of-Domain).
  6. **SingleEq** (Koncel-Kedziorski et al., 2015) — Grade-school algebra problems (Out-of-Domain).
- **Rationale Generation:** Qwen2.5-Math-7B-Instruct was employed to generate high-quality chain-of-thought rationales.
- **Hardware:** Single NVIDIA Tesla H100 GPU with 80GB VRAM.

---

### Table 1: Statistics of Mathematical Reasoning Datasets

| Dataset | Split Type | Answer Type | # Train Samples | # Test Samples | Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | In-Domain | Number | 7,473 | 1,319 | Grade school math |
| **MAWPS** | In-Domain | Number | 1,987 | 398 | Arithmetic & algebra |
| **AQuA** | In-Domain | Multiple Choice | 254 | 254 | Algebraic word problems |
| **MultiArith** | Out-of-Domain | Number | - | 600 | Multi-step arithmetic |
| **AddSub** | Out-of-Domain | Number | - | 395 | Addition / Subtraction |
| **SingleEq** | Out-of-Domain | Number | - | 508 | Single-equation algebra |

---

### 4.2 Main Results on Mathematical Reasoning

> **Figure 3: Performance Comparison Across Mathematical Benchmarks**
> 
> | Method | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
> | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
> | **Huginn** (Base) | 43.59 | 71.85 | 27.95 | 79.83 | 71.90 | 76.97 | **62.02** |
> | **Huginn-SFT** (LoRA) | 49.43 | 78.15 | 30.71 | 87.17 | 74.68 | 80.31 | **66.74** |
> | **MoDr (Ours)** | **49.89** | **80.67** | **33.07** | **91.17** | **79.24** | **81.30** | **69.22** |
> | *Gain over Base* | *+6.30%* | *+8.82%* | *+5.12%* | *+11.34%* | *+7.34%* | *+4.33%* | ***+7.20%*** |
> | *Gain over SFT* | *+0.46%* | *+2.52%* | *+2.36%* | *+4.00%* | *+4.56%* | *+0.99%* | ***+2.48%*** |

---

### 4.3 ABLATION STUDY

#### Impact of Router

### Table 2: Performance Comparison Across Router Configurations

| Configuration | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **No Router (Random Branch Selection)** | 50.72 | 79.41 | 31.89 | 90.17 | 75.70 | 76.57 | **67.41** |
| **MoDr w/o Router at Inference** | 48.60 | 77.73 | 29.53 | 88.00 | 74.68 | 78.35 | **66.41** |
| **MoDr w/ Router (Ours)** | **49.89** | **80.67** | **33.07** | **91.17** | **79.24** | **81.30** | **69.22** |

#### Impact of Single Branch

### Table 3: Performance of Individual Isolated Branches vs Dynamic Routing

| Method | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MoDr (Dynamic Routing)** | **49.89** | **80.67** | 33.07 | 91.17 | **79.24** | **81.30** | **69.22** |
| ↳ **Branch-(1) Only** | 50.19 | 75.63 | **35.43** | 91.67 | 71.39 | 75.98 | **66.72** |
| ↳ **Branch-(2) Only** | 48.07 | 79.37 | 34.65 | 89.17 | 75.19 | 80.71 | **67.86** |
| ↳ **Branch-(3) Only** | 47.84 | 80.25 | 28.35 | **92.50** | 77.97 | 80.71 | **67.94** |
| ↳ **Branch-(4) Only** | 49.66 | 74.37 | 26.77 | 86.83 | 70.13 | 78.15 | **64.32** |
| ↳ **Avg.Br-(1–4)** | 48.94 | 77.41 | 31.30 | 90.04 | 73.67 | 78.89 | **66.71** |

#### Impact of Load Balance
An unbalanced branch load can lead to routing collapse (Shazeer et al., 2017). To evaluate our load balancing strategy governed by update rate $\eta$, we define **Balance Entropy**:

$$ H_{\text{balance}} = -\sum_{br \in \text{Unique}(\Omega)} \frac{\text{Count}(br)}{|\Omega|} \log_2 \frac{\text{Count}(br)}{|\Omega|} \tag{20} $$

where $\Omega$ is the set of branches selected by the router within a batch.
- **$\eta = 0$ (Without Load Balance):** Router collapses to a limited subset of branches ($H_{\text{balance}} \to 0$), achieving **68.66%** average accuracy.
- **$\eta = 0.001$ (With Load Balance):** Uniform distribution maintained across all branches ($H_{\text{balance}} \approx 1.85$), achieving **69.22%** average accuracy.

---

### 4.4 SENSITIVITY ANALYSIS OF RECURRENT BRANCH NUMBERS

> **Figure 5: Performance Across Number of Recurrent Branches**
> - **1 Branch:** 67.0%
> - **2 Branches:** 67.8%
> - **3 Branches:** 68.4%
> - **4 Branches:** **69.22%**
> - **8 Branches:** 69.4%
> - **12 Branches:** 69.5%
> 
> *Observation:* Performance scales positively with branch count up to 4 branches, beyond which gains begin to plateau, confirming $N=4$ as the optimal computational tradeoff.

---

### 4.5 QUANTITATIVE ANALYSIS OF BRANCH ROUTING
Frequency analysis across six mathematical benchmarks (Figure 6) reveals distinct branch specialization:
- **Branch-2 ("Generalist"):** Handles 35–45% of routing consistently across all datasets.
- **Branch-3 ("Arithmetic Specialist"):** Activated at high frequency (46.72% on MultiArith, 38.5% on MAWPS).
- **Branch-4 ("Multiple-Choice Specialist"):** Highly engaged on option-selection tasks (33.16% on AQuA).

---

### 4.6 CASE STUDY
Detailed token trajectory logs demonstrate that MoDr dynamically invokes distinct branches during generation (e.g. Branch 1/2 for context encoding, Branch 3 for intermediate calculation, Branch 4 for conclusion formulation). See Appendix A.3.

---

## 5 CONCLUSION

In this paper, we introduce the Mixture-of-Depth-Recurrent (MoDr) Transformer, a novel dynamic routing framework that advances the depth-recurrent Huginn model. The vanilla Huginn model’s reasoning flexibility is constrained by its reliance on a single, chain-like propagation mechanism within the rumination recurrent module. MoDr addresses this limitation by incorporating multiple LoRA branches and employing a hard-gate router to dynamically select the most appropriate branch for next-token prediction. Extensive experiments across a diverse set of mathematical and commonsense reasoning benchmarks demonstrate that MoDr significantly improves upon the performance of the existing Huginn model while incurring negligible computational overhead.

### 5.1 LIMITATIONS & FUTURE WORK
MoDr offers a dynamic multi-branch framework for the depth-recurrent Huginn model (Geiping et al., 2025), designed to enhance the exploration capability and adaptivity of its rumination recurrent module (Loop) within the latent space. By leveraging LoRAs as distinct branches, our approach avoids significant computational overhead. However, for practical deployment, MoDr necessitates an efficient KV cache strategy, which remains a key challenge and a primary direction for future work. Inspired by (Geiping et al., 2025; Bae et al., 2025), we identify two promising solutions: (1) caching KV pairs from the most recent $k$ recurrent iterations under a fixed budget, or (2) caching the initial KV pairs and sharing them across all recurrent branches for subsequent reasoning steps.

---

## ACKNOWLEDGMENTS

We thank anonymous reviewers for their valuable and insightful feedback. The computational resources for this project were supported by DataCanvas and University of Southampton.

---

## REFERENCES

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). GPT-4 technical report. *arXiv preprint arXiv:2303.08774*.
2. Bae, S., Kim, H., & Lee, S. (2025). Fast and memory-efficient recurrent transformers via token-level KV sharing. *arXiv preprint*.
3. Bisk, Y., Zellers, R., Le Bras, R., Gao, J., & Choi, Y. (2020). PIQA: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing*, 34(05):7432–7439.
4. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. (2021). Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.
5. Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). Universal transformers. In *International Conference on Learning Representations (ICLR)*.
6. DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, X., Zhang, R., Xu, R., Zheng, R., et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint arXiv:2501.12948*.
7. Du, Y., et al. (2025). Super-GPQA: A challenging graduate-level reasoning benchmark across 285 disciplines. *arXiv preprint*.
8. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39.
9. Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., & Neubig, G. (2023). PAL: Program-aided language models. In *International Conference on Machine Learning (ICML)*, pp. 10764–10799.
10. Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). Scaling up test-time compute with latent reasoning: A recurrent depth approach. *arXiv preprint arXiv:2502.05171*.
11. Goyal, A., Lamb, A., Hoffmann, J., Sodhani, S., Levine, S., Bengio, Y., & Schölkopf, B. (2021). Recurrent independent mechanisms. In *International Conference on Learning Representations (ICLR)*.
12. Guo, D., Zhu, Q., Yang, D., Xie, Z., Dong, K., Zhang, W., Chen, G., Bi, X., et al. (2025). DeepSeek-V3 technical report. *arXiv preprint arXiv:2412.19437*.
13. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring mathematical problem solving with the MATH dataset. In *NeurIPS Datasets and Benchmarks*.
14. Hosseini, M. J., Hajishirzi, H., Etzioni, O., & Kushman, N. (2014). Learning to solve arithmetic word problems with verb categorization. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 523–533.
15. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.
16. Huang, B., Geng, Z., & Kolter, J. Z. (2025). Equilibrium reasoners: Learning attractors enables scalable reasoning. *arXiv preprint*.
17. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35:22199–22213.
18. Koncel-Kedziorski, R., Hajishirzi, H., Sabharwal, A., Etzioni, O., & Cohen, S. (2015). Parsing algebraic word problems into equations. *Transactions of the Association for Computational Linguistics (TACL)*, 3:585–597.
19. Koncel-Kedziorski, R., Roy, S., Amini, A., Kushman, N., & Hajishirzi, H. (2016). MAWPS: A math word problem repository. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, pp. 1152–1157.
20. Labovich, A. (2025). Stability and generalization in looped transformers. *arXiv preprint*.
21. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2020). GShard: Scaling giant models with conditional computation and automatic partitioning. In *International Conference on Learning Representations (ICLR)*.
22. Ling, W., Yogatama, D., Dyer, C., & Blunsom, P. (2017). Program induction by rationale generation: Learning to solve algebra word problems. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 158–167.
23. Roy, S., & Roth, D. (2016). Solving general arithmetic word problems. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, pp. 1743–1752.
24. Saunshi, N., Dikkala, N., Li, Z., Kumar, S., & Reddi, S. J. (2025). Reasoning with latent thoughts: On the power of looped transformers. *arXiv preprint arXiv:2502.17416*.
25. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations (ICLR)*.
26. Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., & He, Y. (2025). CODI: Compressing chain-of-thought into continuous space via self-distillation. *arXiv preprint arXiv:2502.21074*.
27. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
28. Wang, Y., et al. (2024). Overcoming conflicting gradients in auxiliary-loss-based MoE training. *ICML*.
29. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022b). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35:24824–24837.
30. Zheng, C., et al. (2024). CodeFeedback: High-quality instruction dataset for code generation. *arXiv preprint*.

---

## APPENDIX

### A.1 ADDITIONAL DISCUSSIONS

#### A.1.1 Latent Space vs. Text Space Reasoning
Verbalized CoT generates reasoning tokens sequentially in the discrete token space. Although interpretable, this process suffers from two major limitations:
1. **Computational Cost:** Generating lengthy reasoning chains (often thousands of tokens) linearly increases test-time FLOPs and KV cache consumption.
2. **Error Cascading:** Once an incorrect reasoning token is generated, subsequent tokens inherit and amplify the error due to autoregressive conditioning.

In contrast, latent reasoning in continuous activation space (conducted in looped/recurrent Transformers) allows the model to refine representations without generating text tokens, preserving alternative hypotheses in superposition and enabling dynamic depth allocation.

#### A.1.2 Soft MoE vs. Hard MoE
- **Soft MoE:** Computes weighted mixtures across all $N$ expert branches. This forces every token to execute all $N$ forward passes, multiplying computation by $O(N)$ and eliminating test-time efficiency gains.
- **Hard MoE (MoDr):** Uses Top-1 (or Top-2) hard-gate routing to dispatch tokens to specific LoRA branches. This maintains active computational cost at $O(1)$ while unlocking diverse specialization.

---

### A.2 EXPERIMENTS FOR COMMONSENSE REASONING

#### A.2.1 Experimental Setup
We evaluate MoDr on six standard commonsense reasoning benchmarks:
1. **PIQA** (Bisk et al., 2020) — Physical interaction QA.
2. **HellaSwag** (Zellers et al., 2019) — Situational continuation.
3. **WinoGrande** (Sakaguchi et al., 2021) — Coreference resolution.
4. **ARC-Easy (ARC-E)** (Clark et al., 2018) — Elementary science questions.
5. **ARC-Challenge (ARC-C)** (Clark et al., 2018) — Complex science questions.
6. **OpenBookQA (OBQA)** (Mihaylov et al., 2018) — Science facts with open book.

Fine-tuning used the `commonsense_170k` instruction tuning dataset.

---

### Table 4: Statistics of Commonsense Reasoning Datasets

| Dataset | Answer Type | # Train Samples | # Test Samples | Description / Domain |
| :--- | :--- | :--- | :--- | :--- |
| **PIQA** | Multiple Choice (2 Options) | 16.1K | 1,838 | Physical commonsense QA |
| **HellaSwag** | Multiple Choice (4 Options) | 39.9K | 10,042 | Situational continuation |
| **WinoGrande** | Multiple Choice (2 Options) | 63.2K | 1,267 | Coreference resolution |
| **ARC-Easy (ARC-E)** | Multiple Choice (4 Options) | 1.1K | 2,376 | Elementary science QA |
| **ARC-Challenge (ARC-C)** | Multiple Choice (4 Options) | 2.3K | 1,172 | Complex science QA |
| **OpenBookQA (OBQA)** | Multiple Choice (4 Options) | 5.0K | 500 | Multi-hop science facts |

---

#### A.2.2 Results and Analysis

> **Figure 7: Performance Comparison on Commonsense Reasoning Benchmarks**
> 
> | Method | PIQA | HellaSwag | WinoGrande | ARC-E | ARC-C | OBQA | **Average** |
> | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
> | **Huginn** (Base) | 75.79 | 64.67 | 57.77 | 69.49 | 37.71 | 37.60 | **57.17** |
> | **Huginn-SFT** | 79.11 | 80.15 | 74.35 | 84.05 | 68.26 | 74.40 | **76.86** |
> | **MoDr (Ours)** | **79.71** | **80.98** | **75.93** | **85.52** | **70.56** | **78.40** | **78.38** |
> | *Gain over Base* | *+3.92%* | *+16.31%* | *+18.16%* | *+16.03%* | *+32.85%* | *+40.80%* | ***+21.21%*** |
> | *Gain over SFT* | *+0.60%* | *+0.83%* | *+1.58%* | *+1.47%* | *+2.30%* | *+4.00%* | ***+1.52%*** |

---

### A.3 CASE STUDIES ON DYNAMIC TOKEN ROUTING

Figures 8, 9, and 10 illustrate the dynamic routing trajectories of tokens across diverse reasoning tasks:
- **Case 1 (Arithmetic Problem Solving):** When processing problem setup tokens ("*A store sells 48 clips in April...*"), the router selects Branch 1 and Branch 2 to encode linguistic entities. When transitioning to calculation ("*48 / 2 = 24*", "*48 + 24 = 72*"), routing switches immediately to Branch 3 (Arithmetic specialist). For the final response formatting ("*Final Answer: \boxed{72}*"), Branch 4 is engaged.
- **Case 2 (Algebraic Deduction):** On multi-variable equations, Branch 3 and Branch 2 alternate across iterative steps, demonstrating collaborative "relay" reasoning across latent depth.

---

### A.4 EXPERIMENTS ON CODE GENERATION

To evaluate code generation capabilities, models were fine-tuned on the `CodeFeedback-Filtered-Instruction` dataset and evaluated on **HumanEval** and **MBPP** with zero-shot greedy Pass@1:

> **Figure 11: Pass@1 on Code Generation Benchmarks**
> 
> | Method | HumanEval (Pass@1) | MBPP (Pass@1) | **Average** |
> | :--- | :--- | :--- | :--- |
> | **Huginn** (Base) | 12.20% | 15.18% | **13.69%** |
> | **Huginn-SFT** | 24.39% | 26.46% | **25.43%** |
> | **MoDr (Ours)** | **26.83%** | **28.02%** | **27.43%** |
> | *Gain over Base* | *+14.63%* | *+12.84%* | ***+13.74%*** |
> | *Gain over SFT* | *+2.44%* | *+1.56%* | ***+2.00%*** |

---

### A.5 COMPARISON WITH STANDARD TRANSFORMER MODELS

### Table 5: MoDr vs Standard Transformer Baselines on Mathematical Reasoning

| Model | Model Size | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Math-1.5B** | 1.5B | 52.46 | 82.66 | 35.83 | 94.67 | 82.53 | 84.84 | **72.17** |
| **DeepSeekMath-Base-7B** | 7.0B | 48.90 | 79.40 | 31.89 | 89.17 | 78.48 | 80.51 | **68.06** |
| **LLaMA-3-8B** | 8.0B | 50.04 | 79.90 | 30.71 | 89.67 | 79.49 | 81.30 | **68.52** |
| **Huginn** (Base) | 3.5B | 43.59 | 71.85 | 27.95 | 79.83 | 71.90 | 76.97 | **62.02** |
| **Huginn-SFT** | 3.5B | 49.43 | 78.15 | 30.71 | 87.17 | 74.68 | 80.31 | **66.74** |
| **MoDr (Ours)** | 3.5B | **49.89** | **80.67** | **33.07** | **91.17** | **79.24** | **81.30** | **69.22** |

---

### A.6 EXPANDED RESULTS ON MATHEMATICAL REASONING

### Table 6: Comparison of MoDr with Prompting and Fine-Tuning Baselines

| Method | Type | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8-Shot CoT** | Prompting | 37.23 | 72.69 | 27.95 | 76.33 | 74.18 | 73.03 | **60.24** |
| **8-Shot CoT-MV** | Majority Vote | 37.76 | 79.41 | 29.53 | 81.17 | 79.49 | 74.02 | **63.56** |
| **Full Fine-Tuning (Full-FT)** | Fine-Tuning | 6.67 | 50.00 | 6.69 | 78.83 | 65.32 | 47.83 | **42.56** |
| **MoDr (Ours)** | LoRA + MoE | **49.89** | **80.67** | **33.07** | **91.17** | **79.24** | **81.30** | **69.22** |

---

### A.7 EXPANDED RESULTS OF CHALLENGING REASONING BENCHMARK: SUPER-GPQA

### Table 7: Performance on Super-GPQA Benchmark Across 285 Graduate Disciplines

| Method | Super-GPQA Score (%) |
| :--- | :--- |
| **Random Guess** | 10.59% |
| **Huginn** (Base) | 10.63% |
| **Huginn-SFT** | 14.07% |
| **MoDr (Ours)** | **14.40%** |

---

### A.8 ANALYSIS OF TOP-K ROUTER

### Table 8: Impact of Top-K Routing Configurations

| Method | Top-K | GSM8K (ID) | MAWPS (ID) | AQuA (ID) | MultiArith (OOD) | AddSub (OOD) | SingleEq (OOD) | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MoDr** | **Top-1** | 49.89 | 80.67 | **33.07** | 91.17 | 79.24 | 81.30 | **69.22** |
| **MoDr** | **Top-2** | **50.27** | **84.45** | 28.35 | **96.67** | **85.82** | **84.06** | **71.60** |

---

### A.9 PERFORMANCE COMPARISON WITH SAME TRAINING DURATION

### Table 9: Performance Comparison with Huginn-SFT Under Identical Training Duration

| Method | Top-K | MAWPS | AQuA | MultiArith | AddSub | SingleEq | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Huginn-SFT** | - | 80.25 | 31.50 | 93.17 | 77.97 | 81.30 | **72.84** |
| **MoDr (Ours)** | **1** | 80.67 | **33.07** | 91.17 | 79.24 | 81.30 | **73.09** |
| **MoDr (Ours)** | **2** | **84.45** | 28.35 | **96.67** | **85.82** | **84.06** | **75.87** |

---

### A.10 DETAILS FOR EXPERIMENTS / COMPUTATIONAL EFFICIENCY

### Table 10: Resource Requirements and Computational Efficiency Comparison

| Metric | Huginn (Base) | Huginn-SFT | MoDr (Ours) |
| :--- | :--- | :--- | :--- |
| **Total Resident Parameters** | 3.56B | 3.56B | 3.56B |
| **Trainable Parameters** | 3.56B (N/A) | 2.03M (0.057%) | 8.13M (0.228%) |
| **Active Compute (TFLOPs)** | 35.00 | 35.55 | 35.60 (+0.14%) |
| **Training VRAM** | N/A | 54 GB | 54 GB |
| **Deployment Memory** | 13 GB | 13 GB | 13 GB |
| **Training Duration (H100)** | N/A | 3h 58min | 16h 26min |
| **Inference Generation Speed** | 15 tokens/s | 14 tokens/s | 13 tokens/s |

---

### A.11 EVALUATION DETAILS & PROMPT TEMPLATES

All evaluation experiments used temperature = 0.0001, max_tokens = 1024, and recurrent loop steps $T = 16$.

```markdown
### Evaluation Prompt for Mathematical Reasoning Tasks
System: Please reason step by step, and put your final answer within \boxed{}.
User: {Question}
Assistant:

### Evaluation Prompt for Commonsense Reasoning Tasks
System: Below is an instruction that describes a task. Write a response that appropriately completes the request.
User:
### Instruction: {Instruction}
### Response:
Assistant:

### Evaluation Prompt for Code Generation Tasks (HumanEval / MBPP)
System: You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
User:
// For HumanEval
### Instruction: Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
Function signature: {Example}
### Response:

// For MBPP
### Instruction: {Test}
Example Test Cases: {Test List}
### Response:
Assistant:
```

---

### A.12 FINE-TUNING & SOLUTION EXAMPLES

#### A.12.1 Mathematical Reasoning Examples

**Example 1 (Multi-Step Arithmetic):**
*Question:* A store sells 48 paper clips in April. In May, it sells half as many paper clips as in April. How many paper clips did the store sell in April and May combined?

*Step-by-Step Solution:*
1. Clips sold in April = 48
2. Clips sold in May = 48 / 2 = 24
3. Total clips sold = Clips in April + Clips in May = 48 + 24 = 72
4. Final Answer: `\boxed{72}`

**Example 2 (Algebraic Multiplication):**
*Calculate:* $9.95 \times 20$
$$ 9.95 \times 20 = (10 - 0.05) \times 20 = (10 \times 20) - (0.05 \times 20) = 200 - 1 = 199 $$
Final Answer: `\boxed{199}`

**Example 3 (Decomposition):**
*Calculate:* $27 \times 16$
$$ 27 \times 16 = 27 \times (10 + 6) = (27 \times 10) + (27 \times 6) $$
$$ 27 \times 10 = 270 $$
$$ 27 \times 6 = 162 $$
$$ 270 + 162 = 432 $$
Final Answer: `\boxed{432}`

#### A.12.2 Commonsense Reasoning Example
*Instruction:* Please choose the correct answer to the question: Which factor will most likely cause a person to develop a fever?
- A: a leg muscle spasm
- B: a bacterial infection
- C: a reduced heart rate
- D: an elevated blood pressure

*Response:* B (A bacterial infection triggers immune release of pyrogens, inducing physiological temperature increase).

#### A.12.3 Code Generation Example
*Instruction:* Write a function `has_close_elements(numbers: List[float], threshold: float) -> bool` that checks if any two distinct elements in the list are closer to each other than the given threshold.

```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
```
