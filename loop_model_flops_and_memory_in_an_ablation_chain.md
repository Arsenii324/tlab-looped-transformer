Contents

- [Introduction](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#introduction)
- [Setup and Measurement Contract](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#setup-and-measurement-contract)

- [Defining the FLOPs Cost Model](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-flops-cost-model)
- [Defining the Memory Cost Model](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-memory-cost-model)
- [The Ablation Chain](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#the-ablation-chain)

- [V1. Baseline: non-shared + final loss](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v1-baseline-non-shared--final-loss)
- [V2. + shared weights](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v2--shared-weights)
- [V3. + per-step losses](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v3--per-step-losses)
- [V4. + detach each step under end-of-rollout backward](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v4--detach-each-step-under-end-of-rollout-backward)
- [V5. + instant update](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v5--instant-update)
- [V6. + internal truncation inside the cell](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v6--internal-truncation-inside-the-cell)
- [V7. + gradient checkpointing](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v7--gradient-checkpointing)
- [Toy Experiment Check](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#toy-experiment-check)

- [Benchmark setup](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#benchmark-setup)
- [Results](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#results)
- [Key Takeaways](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#key-takeaways)
- [Future Topics](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#future-topics)

[Home](https://huskydoge.github.io/husky-blog/) » [Posts](https://huskydoge.github.io/husky-blog/posts/)

# Loop-Model FLOPs and Memory in an Ablation Chain

April 19, 2026 · 40 min · 8470 words · Benhao Huang ·  | [Suggest Changes](https://github.com/huskydoge/husky-blog/edit/main/content/posts/recursive_models/loop-cost/post.en.md)

#### ▾ Collapsible Section: Table of Contents

- [Introduction](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#introduction)
- [Setup and Measurement Contract](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#setup-and-measurement-contract)

- [Defining the FLOPs Cost Model](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-flops-cost-model)
- [Defining the Memory Cost Model](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-memory-cost-model)
- [The Ablation Chain](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#the-ablation-chain)

- [V1. Baseline: non-shared + final loss](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v1-baseline-non-shared--final-loss)
- [V2. + shared weights](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v2--shared-weights)
- [V3. + per-step losses](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v3--per-step-losses)
- [V4. + detach each step under end-of-rollout backward](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v4--detach-each-step-under-end-of-rollout-backward)
- [V5. + instant update](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v5--instant-update)
- [V6. + internal truncation inside the cell](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v6--internal-truncation-inside-the-cell)
- [V7. + gradient checkpointing](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#v7--gradient-checkpointing)
- [Toy Experiment Check](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#toy-experiment-check)

- [Benchmark setup](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#benchmark-setup)
- [Results](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#results)
- [Key Takeaways](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#key-takeaways)
- [Future Topics](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#future-topics)

![Loop-model FLOPs and memory teaser schematic](./Loop-Model FLOPs and Memory in an Ablation Chain _ Husky'Log_files/teaser.png)

## Introduction

Loop models are becoming popular lately, with exciting results [[1,2,3,4,5]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
. Once we decide to reuse the same block across multiple layers, however, one practical question becomes unavoidable: **what does the loop cost during training?**

That is the question of this post. Loop models now come with a growing collection of training tricks and schedules, but those choices are easy to blur together if we describe them all under the loose label of “recurrence.” In this post, I build a clean ablation chain over the most common ingredients: non-shared versus shared weights, terminal loss versus per-step losses, outer-step `detach`, instant update, internal truncation inside the cell, and gradient checkpointing.

**Disclaimer**. I am **not** trying to settle which training strategy gives the best downstream performance. Different strategies change not only cost, but also the learning algorithm itself, so accuracy comparisons deserve a separate discussion. In this essay, I focus on the **cost analysis**.

## Setup and Measurement Contract

A common loop model takes the following structure.

Write the common outer loop as

$$ h_{t}=f_{\thet a}(h_{t-1},x_{t}), t=1,\dot s ,T, $$

where  $h_{0}$  is an initial state that is usually not parameterized.

Notation Convention

Throughout the post,

- **cell** means one call to the repeated computation  $f_{\thet a}$  in the **outer** loop. When that cell contains its own solver, I call those internal iterations **inner loops**.
- **NFE** means the **number of forward applications of the repeated block that define the effective depth of the computation**. In the simple outer-loop picture above, that is one count per outer-step cell call. In the toy benchmark later, where each outer-step cell contains an inner refinement loop, I count those inner refinements toward NFE, because they are the actual repeated block applications that make up the logical depth.
- **checkpointing** means gradient checkpointing. I keep the full term when first introducing it, and then shorten it to checkpointing.

For the cost story, two Jacobian objects matter:

$$ \mathb f{J_{t}}=\fra c{∂h_{t}}{∂h_{t-1}}, \mathb f{B_{t}}=\fra c{∂h_{t}}{∂\thet a_{t}}\tex t{or}\mathb f{B_{t}}=\fra c{∂h_{t}}{∂\thet a}, $$

depending on whether the weights are non-shared or shared.

A compact schematic for the final step is:

> **[Architecture Schematic]** *Loop-model cost schematic showing the final-step chain h_{T-1} to f_theta to h_T to ell_T, a top temporal arrow J_T equal to partial h_T over partial h_{T-1}, and a parameter arrow B_T equal to partial h_T over partial theta from h_T down to theta.*
> *(Text in diagram: ⋯JT= ∂hT/ ∂hT−1hT−1fθhTℓTBT= ∂hT/ ∂θθ)*

In this picture, the uppercase index  $T$  simply marks the **last** step of the rollout. The blue top arrow is therefore the last temporal Jacobian,

$$ J_{T}=\fra c{∂h_{T}}{∂h_{T-1}}, $$

and the red downward arrow is the last-step local parameter map,

$$ B_{T}=\fra c{∂h_{T}}{∂\thet a_{T}}\tex t{or}B_{T}=\fra c{∂h_{T}}{∂\thet a}, $$

depending on whether the weights are non-shared or shared.

With that notation in place, we can finally state the measurement contract used in the rest of the post. Throughout, we focus on the costs during one **training interval/optimizer interval**, namely the computation between two model updates. Unless stated otherwise, **Variants 1-4 all use the same schedule**: run the full  $T$ -step outer rollout, accumulate all outer losses, launch one backward pass, and then take one optimizer step. **Variants 5-7 intentionally change that contract**: once instant update appears, there is one optimizer step **per outer step**, which is the schedule used in HRM and TRM-style training setups [[1,5]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
.

This distinction matters because the optimizer interval determines what counts as one training interval, which in turn changes both the backward graph and the lifetime of saved tensors.

Throughout the post, the two main cost types refer to different parts of this graph:

- **Symbolic FLOPs cost** counts backward-side training arithmetic at fixed forward work: propagating a state gradient through  $\mathb f{J_{t}}$ , forming parameter-gradient contributions through  $\mathb f{B_{t}}$ , accumulating shared gradients, and applying the optimizer update.
- **Memory cost** counts tensors that remain alive across the forward/backward boundary.
- **Parameter-side memory** is also real, and I account for it separately through parameter tensors, gradient buffers, and optimizer states.

### Defining the FLOPs Cost Model

We define the following local costs:

- $c_{ℓ}$ : cost of one local loss backward  $∂ℓ_{t}/∂h_{t}$ ; for example, backward through a softmax-cross-entropy head or an MSE decoder attached to  $h_{t}$ .
- $e_{J}$ : cost of obtaining or evaluating the local Jacobian operator  $\mathb f{J_{t}}$ .
- $m_{J}$ : cost of multiplying the incoming state gradient by  $\mathb f{J_{t}}$ .
- $e_{B}$ : cost of obtaining or evaluating the local Jacobian operator  $\mathb f{B_{t}}$ .
- $m_{B}$ : cost of multiplying the incoming state gradient by  $\mathb f{B_{t}}$ .
- $c_{h}$ : cost of one hidden-state gradient addition; for example, adding the current-step local gradient to the future-to-past contribution in a recursion like  $a_{t}=\delt a_{t}+a_{t+1}J_{t+1}$ .
- $c_{\thet a}$ : cost of one accumulation into the shared-gradient buffer for one step-local parameter block; under tensor-level accounting, multiply by the number of tensors in that block.
- $c_{u}$ : cost of updating one step-local parameter block once in the optimizer step; under tensor-level accounting, multiply by the number of tensors in that block. For example, SGD applies  $W\get s W-ηG$ , while Adam-style updates also touch moment buffers.

Because the Jacobian-evaluation term and the Jacobian-product term usually appear together, define the grouped costs

$$ c_{J}=e_{J}+m_{J}, c_{B}=e_{B}+m_{B}. $$

This split is just a convenient decomposition rather than a claim that autograd materializes or pays for two fully separate Jacobian constructions. In real kernels, parts of the local derivative preparation may be shared or fused across the temporal and parameter VJP paths.

I keep the  $e_{∗}$  /  $m_{∗}$  split for interpretation, but most formulas below use the compact forms  $c_{J}$  and  $c_{B}$ .

Here the incoming row adjoint is

$$ a_{t}=\fra c{dL}{dh_{t}}. $$

The right-boundary condition is whichever loss touches the final state. For a terminal-loss rollout,

$$ a_{T}=\fra c{∂ℓ(h_{T})}{∂h_{T}}, $$

while for a per-step-loss rollout,

$$ a_{T}=\delt a_{T}=\fra c{∂ℓ_{T}}{∂h_{T}}. $$

This does **not** mean explicitly materializing full Jacobian matrices. It is only a convenient decomposition between local Jacobian evaluation and the arithmetic driven by the incoming adjoint; in real autograd code, these substeps are often fused inside one backward kernel.

Both  $c_{\thet a}$  and  $c_{u}$  scale with parameter shape. The difference is conceptual:

- $c_{\thet a}$  is one shared-gradient accumulation for one step-local parameter block;
- $c_{u}$  is one optimizer update of one step-local parameter block.

For SGD,  $c_{\thet a}$  and  $c_{u}$  are often of similar order. For Adam-style optimizers,  $c_{u}$  is typically larger because the optimizer also updates moment buffers and applies extra elementwise operations.

### Defining the Memory Cost Model

For memory, it helps to separate activation-side effects from parameter-side effects. Start from

$$ M_{\tex t{total}}=M_{\tex t{act}}+M_{\tex t{param}}+M_{\tex t{grad}}+M_{\tex t{opt}}. $$

The main formulas below focus on the activation term  $M_{\tex t{act}}$ , because `detach`, checkpointing, and internal truncation primarily change that term.[^1] Parameter-side memory still matters, so I track it separately through parameter tensors, gradient buffers, and optimizer state.

Define:

- $p_{\thet a}$ : storage of one parameter block.
- $p_{g}$ : storage of one parameter-gradient buffer.
- $p_{\tex t{opt}}$ : storage of optimizer state for one parameter block. For plain SGD without momentum this can be  $0$ ; for momentum SGD it is often about one extra tensor; for Adam-style methods it is often about two extra tensors.

For shared and non-shared parameterizations, the parameter-side memory is

$$ M_{\tex t{param-side}}^{\tex t{shared}}=p_{\thet a}+p_{g}+p_{\tex t{opt}}, M_{\tex t{param-side}}^{\tex t{non-shared}}(T)=T(p_{\thet a}+p_{g}+p_{\tex t{opt}}). $$

These parameter-side formulas refer to the recurrent block and whatever head is included in the modeled training cell. In larger language-model style systems, prelude and coda parameters contribute additional mostly constant terms relative to the outer-loop ablation chain here.

Now define the activation-side quantities:

- $a_{f}$ : activation memory retained by one iterative cell  $f$  under ordinary autograd. For the affine-tanh reference cell  $h_{t}=\tan h(W_{h}h_{t-1}+W_{x}x_{t}+b)$ , this includes the boundary hidden state, the preactivation  $z_{t}=W_{h}h_{t-1}+W_{x}x_{t}+b$ , and any normalization or mask tensors saved for backward. If you want finer resolution, write  $a_{f}=a_{h}+a_{\tex t{int}}$ , where  $a_{h}$  is the boundary hidden state and  $a_{\tex t{int}}$  are the internal saved tensors of that step.
- $a_{ℓ}$ : activation memory retained by one loss/head branch; for example, logits  $o_{t}=Uh_{t}$  together with softmax or decoder-side saved tensors.
- $a_{f}^{\tex t{ckpt}}$ : activation memory retained by one iterative cell under checkpointing, with  $a_{f}^{\tex t{ckpt}}\l e a_{f}$ , typically with strict inequality in the intended checkpointed settings.
- $r_{\tex t{ckpt}}$ : extra recomputation FLOPs induced by checkpointing one iterative cell during backward; for that same affine-tanh cell, this is the cost of rerunning the affine map and nonlinearity to reconstruct dropped intermediates.
- $e_{B}^{\tex t{trunc}}$ : cost of obtaining or evaluating the truncated local Jacobian operator for the parameter path when gradients are truncated **inside** the cell.
- $m_{B}^{\tex t{trunc}}$ : cost of multiplying the incoming gradient by that truncated local parameter Jacobian operator.
- $a_{f}^{\tex t{trunc}}$ : activation memory retained by one iterative cell when gradients are truncated inside the cell.
- $r_{\tex t{ckpt}}^{\tex t{trunc}}$ : extra recomputation FLOPs when checkpointing the remaining differentiable part of an already truncated cell.
- $a_{f}^{\tex t{trunc,ckpt}}$ : activation memory retained by one truncated iterative cell under checkpointing.

For compact formulas in the truncated case, also define

$$ c_{B}^{\tex t{trunc}}=e_{B}^{\tex t{trunc}}+m_{B}^{\tex t{trunc}}. $$

Below, whenever I write  $M_{(⋅)}$  without further qualifiers, I mean the activation-memory term  $M_{\tex t{act}}$ . If you want total training memory, simply add the parameter-side terms above.

Variants 1-4 keep the fixed end-of-rollout schedule introduced at the start of the post. Variants 5-7 intentionally switch to one optimizer step per outer step as soon as the instant-update strategy appears.

Cost LegendCollapse

**FLOPs**

- $c_{ℓ}$ : one local loss backward
- $e_{J},m_{J}$ : split temporal-Jacobian costs; together  $c_{J}=e_{J}+m_{J}$
- $e_{B},m_{B}$ : split local parameter-backward costs; together  $c_{B}=e_{B}+m_{B}$
- $c_{h}$ : one hidden-state gradient addition
- $c_{\thet a}$ : one shared-gradient accumulation
- $c_{u}$ : one optimizer update for one parameter block
- $r_{\tex t{ckpt}}$ : extra recomputation FLOPs from checkpointing
- $c_{B}^{\tex t{trunc}}$ : local parameter-backward cost under internal cell truncation
- $r_{\tex t{ckpt}}^{\tex t{trunc}}$ : extra recomputation FLOPs for checkpointing the remaining differentiable part of a truncated cell

**Memory**

- $a_{f}$ : activation memory of one iterative cell
- $a_{ℓ}$ : activation memory of one loss/head branch
- $a_{f}^{\tex t{ckpt}}$ : activation memory of one checkpointed iterative cell
- $a_{f}^{\tex t{trunc}}$ : activation memory of one iterative cell under internal cell truncation
- $a_{f}^{\tex t{trunc,ckpt}}$ : activation memory of one truncated iterative cell under checkpointing

## The Ablation Chain

From this point on, the cell-local algebra is fixed. Each row changes exactly one training choice. To avoid re-deriving the same local identities every time, keep the [notation above](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-flops-cost-model)

 from [Defining the FLOPs Cost Model](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-flops-cost-model) in view:  $J_{t}$ ,  $B_{t}$ ,  $a_{t}$ , the right-boundary value  $a_{T}$ , and the grouped costs all keep the same meaning. The only thing that changes from row to row is how these local pieces compose across the rollout.

For a terminal-loss rollout,

$$ a_{T}=\fra c{∂ℓ(h_{T})}{∂h_{T}}, a_{t-1}=a_{t}J_{t}, ∇_{\thet a_{t}}L=a_{t}B_{t}. $$

For a per-step-loss rollout, with  $\delt a_{t}=∂ℓ_{t}/∂h_{t}$ ,

$$ a_{T}=\delt a_{T}, a_{t}=\delt a_{t}+a_{t+1}J_{t+1}. $$

The boundary conventions used below are simple:

- **Right boundary:** there is no future term beyond step  $T$ , so every backward recursion starts from  $a_{T}$ .
- **Left boundary:** if  $h_{0}$  is a fixed initial state, we stop after forming  $a_{1}$ ; only if  $h_{0}$  were learnable would we continue one more multiplication to get  $a_{0}=a_{1}J_{1}$ .
- **Degenerate case  $T=1$ :** every temporal term disappears, so any formula with  $(T-1)c_{J}$  or  $(T-1)c_{h}$  collapses to its purely local part.

This is why the formulas below repeatedly show  $T-1$  temporal-Jacobian applications but  $T$  local parameter-gradient terms: a  $T$ -step rollout has  $T-1$  state-to-state edges but  $T$  step-local parameter leaves.

Before walking through the rows, it helps to place the recent loop-model literature into three nearby buckets.

- **HRM-style latent reasoning:** HRM uses repeated supervision segments, detached carried state between segments, parameter updates after each segment, and a one-step gradient approximation for the final inner transition. In the language of this post, it sits closest to the run from `+ detach` to `+ instant update` to `+ internal truncation inside the cell` [[5]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
.
- **TRM-style latent reasoning:** TRM keeps the repeated supervision, detached carried state, and per-segment updates, but backpropagates through the full final recursion process and treats the one-step approximation as a weaker ablation. So it is closer to `+ detach` to `+ instant update`, with full inner-step backward in the last segment rather than Variant 6 style inner truncation [[1]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
.
- **Large-scale looped language models:** recurrent-depth LMs such as Geiping et al.’s recurrent-depth approach and Parcae use a shared recurrent block with a terminal language-model loss, stochastic loop depth, and truncated backpropagation through the main recurrent-depth loop. So they are closest to the shared-weight + terminal-loss setup, plus an extra outer-loop truncation ingredient that is adjacent to, rather than identical with, the seven-row chain below [[2,3]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
.

With that context in mind, let’s start changing one knob at a time.

### V1. Baseline: non-shared + final loss

Variant 1 · non-shared weights + final loss
final loss onlynon-shared θ₁, θ₂, θ₃full temporal path

> **[Architecture Schematic]** *Variant 1 schematic with a full temporal path, one final loss, and three separate parameter leaves.*
> *(Text in diagram: ⋯J₃J₂J₁ℓ_Th0fθ₁h1fθ₂h2fθ₃h3θ1θ2θ3)*

One final loss at h3 sends credit through the whole state chain, and each step owns its own parameter leaf.

**Mental picture:** one blue state chain, one terminal loss, and one separate red parameter leaf at each step.

This untied-depth model is the natural reference point. I keep it because it makes the effect of weight tying completely explicit in the next variant.

Start from the non-shared case:

$$ h_{t}=f_{\thet a_{t}}(h_{t-1},x_{t}), L=ℓ(h_{T}). $$

There are two equivalent ways to write the gradient:

- the **fully expanded chain-rule form**, which makes the dependency on every later step explicit;
- the **adjoint form**, which compresses that suffix product into the state gradient  $a_{t}=dL/dh_{t}$ .

Start with the [notation above](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#defining-the-flops-cost-model)

 fully expanded chain-rule expression:

$$ ∇_{\thet a_{t}}L=\fra c{∂ℓ(h_{T})}{∂h_{T}}(\pro d _{k=t+1}^{T}\fra c{∂h_{k}}{∂h_{k-1}})\fra c{∂h_{t}}{∂\thet a_{t}}. $$

Now define the adjoint at step  $t$  to be the suffix of that expression up to  $h_{t}$ :

$$ a_{t}=\fra c{dL}{dh_{t}}=\fra c{∂ℓ(h_{T})}{∂h_{T}}(\pro d _{k=t+1}^{T}\fra c{∂h_{k}}{∂h_{k-1}}). $$

With that definition, the same gradient becomes the local parameter term hit by the incoming adjoint,

$$ ∇_{\thet a_{t}}L=a_{t}\mathb f{B_{t}}. $$

and the adjoints themselves satisfy the one-step reverse recurrence

$$ a_{t-1}=a_{t}\mathb f{J_{t}}. $$

So the compact notation is not a new approximation or a different derivation. It is just the same expanded chain rule rewritten in terms of the recursively computed state gradient:

$$ ∇_{\thet a_{t}}L=a_{T}(\pro d _{k=t+1}^{T}\mathb f{J_{k}})\mathb f{B_{t}}=a_{t}\mathb f{B_{t}}. $$

Reverse mode touches each kind of edge exactly once:

- $T-1$  uses of  $\mathb f{J_{t}}$ ;
- $T$  uses of  $\mathb f{B_{t}}$ ;
- $T$  optimizer updates, one for each step-local parameter block.

So, under the cost model above,

$$ C_{\tex t{non-shared, final}}(T)=c_{ℓ}+(T-1)c_{J}+Tc_{B}+Tc_{u}. $$

The activation-memory cost is

$$ M_{\tex t{non-shared, final}}(T)=Ta_{f}+a_{ℓ}. $$

All  $T$  iterative cells must keep their step-local activations alive, because the terminal loss sends credit through the entire chain. The gradient graph is therefore a state chain ending in one loss node, with a separate parameter leaf  $\thet a_{t}$  at each step.

### V2. + shared weights

Variant 2 · shared weights + final loss
final loss onlyall red branches merge to one θfull temporal path

> **[Architecture Schematic]** *Variant 2 schematic with a full temporal path, one final loss, and one shared parameter accumulator.*
> *(Text in diagram: ⋯J₃J₂J₁ℓ_Th0fθh1fθh2fθh3shared θ)*

Relative to Variant 1, the blue time path is unchanged; only the parameter side changes from separate leaves to one shared accumulation point.

**What changes:** the blue temporal chain stays the same, but the red stepwise parameter contributions now merge into one shared leaf.

Relative to Variant 1, the blue state recursion is unchanged. Only the red parameter side changes: each step still contributes a local parameter term, but those terms now accumulate into one shared parameter block.

$$ h_{t}=f_{\thet a}(h_{t-1},x_{t}), L=ℓ(h_{T}). $$

The adjoint recursion is therefore exactly the same as in Variant 1:

$$ a_{t-1}=a_{t}\mathb f{J_{t}}. $$

What changes is only how the local parameter contributions are collected. In fully expanded form, the shared-parameter gradient is the sum over all step-local uses of that same parameter block:

$$ ∇_{\thet a}L=\fra c{∂ℓ(h_{T})}{∂h_{T}}\su m _{t=1}^{T}(\pro d _{k=t+1}^{T}\fra c{∂h_{k}}{∂h_{k-1}})\fra c{∂h_{t}}{∂\thet a}. $$

Now reuse the same adjoint definition as in Variant 1,

$$ a_{t}=\fra c{dL}{dh_{t}}=\fra c{∂ℓ(h_{T})}{∂h_{T}}(\pro d _{k=t+1}^{T}\fra c{∂h_{k}}{∂h_{k-1}}), $$

so the shared-parameter gradient compresses to

$$ ∇_{\thet a}L=a_{T}\su m _{t=1}^{T}(\pro d _{k=t+1}^{T}\mathb f{J_{k}})\mathb f{B_{t}}=\su m _{t=1}^{T}a_{t}\mathb f{B_{t}}. $$

This is the key structural difference from Variant 1:

- in Variant 1, each  $a_{t}\mathb f{B_{t}}$  goes to its own parameter block  $\thet a_{t}$ ;
- in Variant 2, the same  $a_{t}\mathb f{B_{t}}$  terms are accumulated into one shared gradient buffer for  $\thet a$ .

Remark: Backpropagation is a dynamic programming algorithmA common mistake is to look at the fully expanded chain rule and conclude that BPTT must be quadratic in  $T$ . That is the cost of naively recomputing suffix chains from scratch. In the rest of this post, every formula refers to the **actual reverse-mode execution cost**. Expanding the chain rule on paper can make BPTT look quadratic in  $T$ , but the formulas above are about the real reverse-mode execution. Each needed local Jacobian operator is applied once per edge in the backward graph; we are **not** re-deriving every suffix chain from scratch.

**Weight tying still gives linear-time BPTT.** Compared with Variant 1, it adds  $T-1$  shared-gradient accumulations and replaces  $T$  separate optimizer updates with one shared update.

Here  $c_{\thet a}$  counts only cross-step accumulation into an already existing shared gradient buffer. The first write is absorbed into the corresponding local parameter-backward term, which is why the accumulation count is  $T-1$  rather than  $T$ .

Hence

$$ C_{\tex t{shared, final}}(T)=c_{ℓ}+(T-1)c_{J}+Tc_{B}+(T-1)c_{\thet a}+c_{u}. $$

The activation-memory cost is unchanged,

$$ M_{\tex t{shared, final}}(T)=Ta_{f}+a_{ℓ}. $$

because tying the parameters does not let step  $t$  reuse the saved activations of step  $t+1$ . The state chain is the same; only the parameter leaves have merged.

![](./Loop-Model FLOPs and Memory in an Ablation Chain _ Husky'Log_files/huggin.png)

*Caption: **Fig. 2:**
Prelude, Recurrent and Coda. Figure from “Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach”*

**This variant is the cleanest symbolic match to the large-scale looped-language-model family.** [Citation: Geiping, *& al.* [2]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#geiping2025recurrentdepth)
; [Citation: Prairie, *& al.* [3]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#prairie2026parcae)
 both follow the same basic prelude  $\t o$  shared recurrent block  $\t o$  coda pattern with one final language-model loss, even though in practice they also introduce sampled loop depth and truncated backpropagation through the main recurrent-depth loop.

### V3. + per-step losses

Variant 3 · shared weights + losses at every step
ℓ1 + ℓ2 + ℓ3shared θfull temporal path

> **[Architecture Schematic]** *Variant 3 schematic with local losses at every step, a full temporal path, and one shared parameter accumulator.*
> *(Text in diagram: ⋯J₃J₂J₁ℓ₁ℓ₂ℓ₃h0fθh1fθh2fθh3shared θ)*

Each state now has its own local loss branch, but the blue cross-time backward path is still intact and all red parameter contributions still merge into one shared θ.

**What changes:** each state now gets its own local loss branch. The temporal chain remains intact.

Relative to Variant 2, add one local loss branch at every step:

$$ h_{t}=f_{\thet a}(h_{t-1},x_{t}), L=\su m _{t=1}^{T}ℓ_{t}(h_{t}). $$

Define the local loss gradient

$$ \delt a_{t}=\fra c{∂ℓ_{t}}{∂h_{t}}. $$

Then the backward recursion becomes

$$ a_{T}=\delt a_{T}, a_{t}=\delt a_{t}+a_{t+1}\mathb f{J_{t+1}}, t=T-1,\dot s ,1. $$

The parameter gradient is still

$$ ∇_{\thet a}L=\su m _{t=1}^{T}a_{t}\mathb f{B_{t}}. $$

Compared with Variant 2, this replaces one terminal loss backward with  $T$  step-local loss backward contributions, so the incremental increase is  $T-1$  additional local loss backward terms plus  $T-1$  hidden-state gradient additions, while keeping the blue temporal path fully alive.

So

$$ C_{\tex t{shared, per-step}}(T)=Tc_{ℓ}+(T-1)c_{J}+(T-1)c_{h}+Tc_{B}+(T-1)c_{\thet a}+c_{u}. $$

The activation-memory cost becomes

$$ M_{\tex t{shared, per-step}}(T)=T(a_{f}+a_{ℓ}). $$

because every step now retains both one iterative cell and one local loss/head branch.

For the next variants, it is useful to separate the cost into local and temporal pieces:

$$ C_{\tex t{local}}=c_{ℓ}+c_{B}+c_{\thet a}, C_{\tex t{temporal}}=c_{J}+c_{h}. $$

Then

$$ C_{\tex t{shared, per-step}}(T)=TC_{\tex t{local}}+(T-1)C_{\tex t{temporal}}-c_{\thet a}+c_{u}. $$

This decomposition makes the next row easy to read: `detach` removes the temporal term and leaves the local term behind.

### V4. + `detach` each step under end-of-rollout backward

Variant 4 · per-step losses + detach each step
ℓ1 + ℓ2 + ℓ3shared θdetach cuts blue edges

> **[Architecture Schematic]** *Variant 4 schematic with per-step losses, detached temporal edges, and a shared parameter accumulator.*
> *(Text in diagram: ⋯detachdetachdetachℓ₁ℓ₂ℓ₃h0fθh1fθh2fθh3shared θ)*

Detach removes only the blue temporal backward edges. The local loss branches and the shared red parameter path remain fully active.

**What changes:** only the blue temporal edges are cut. The gold local losses and the red parameter branches remain.

In this row, I keep the same end-of-rollout execution contract as Variant 3: all detached step losses are accumulated first, and one backward pass is launched only after the full outer rollout.

Relative to Variant 3, cut only the blue temporal edges:

$$ h_{t}=f_{\thet a}(\mathr m{detach}(h_{t-1}),x_{t}), L=\su m _{t=1}^{T}ℓ_{t}(h_{t}). $$

After `detach`, future losses no longer flow backward across time. The recursion collapses to

$$ a_{t}=\delt a_{t}. $$

So the parameter gradient becomes

$$ ∇_{\thet a}L=\su m _{t=1}^{T}\delt a_{t}\mathb f{B_{t}}. $$

This is the key point of the article:

- `detach` removes the blue temporal path. It does **not** remove the gold local loss branches or the red parameter-gradient work.
- Under a fixed end-of-rollout backward, that means **less temporal FLOPs** but still **$T$  detached local graphs kept alive** for backward.

Important Implementation Note

After `detach`, the outer-step graphs are independent. So for the detached per-step objective

$$ L=\su m _{t=1}^{T}ℓ_{t}(h_{t}), h_{t}=f_{\thet a}(\mathr m{detach}(h_{t-1}),x_{t}), $$

there are two execution contracts:

1. accumulate all step losses and call one backward at the end of the rollout;
2. call `loss_t.backward()` after each outer step, accumulate gradients in the shared parameter buffers, and delay `optimizer.step()` until the end of the rollout.

These two contracts produce the same parameter gradient up to floating-point accumulation order, because the step graphs are already disconnected by `detach`. The second contract is usually the more sensible implementation: it releases each detached local graph earlier and reduces the outer-depth activation peak from linear-in- $T$  storage to effectively constant-in- $T$  storage.

I keep the end-of-rollout version here only to keep the comparison clean: in this row, the new change is just that `detach` cuts cross-step gradients, not that backward is called at a different time. In practice, once `detach` is present, streaming backward accumulation is often the better implementation, as the toy experiment section will show.

So, relative to Variant 3, the entire temporal term  $(T-1)C_{\tex t{temporal}}$  disappears, while the local loss branches and red parameter branches remain.

Hence

$$ C_{\tex t{shared, per-step, detach}}(T)=Tc_{ℓ}+Tc_{B}+(T-1)c_{\thet a}+c_{u}. $$

Under the fixed execution schedule above, the activation-memory scaling is still linear in  $T$ :

$$ M_{\tex t{shared, per-step, detach}}(T)=T(\tild e{a}_{f}+\tild e{a}_{ℓ}), \tild e{a}_{f}+\tild e{a}_{ℓ}=\Thet a(a_{f}+a_{ℓ}). $$

Equivalently, the saved FLOPs are exactly the temporal part:

$$ C_{\tex t{shared, per-step}}(T)-C_{\tex t{shared, per-step, detach}}(T)=(T-1)c_{J}+(T-1)c_{h}=(T-1)C_{\tex t{temporal}}. $$

If  $C_{\tex t{local}}$  is large, `detach` may save much less compute than the recurrence diagram first suggests. The memory constant can also shift a bit: after `detach`, each step-local backward no longer needs to send gradients into  $h_{t-1}$ , so an autodiff engine may skip saving some tensors that would only have been needed for that input-gradient path. But the parameter-gradient path is still alive, so most of the step-local state needed for the loss/head backward and the local parameter backward still has to remain. That is why the robust claim here is about scaling rather than exact bytes: **under a fixed end-of-rollout backward, `detach` changes temporal FLOPs while keeping the outer-depth activation scaling linear in  $T$ .**

### V5. + instant update

Variant 5 · per-step losses + detach + instant update
one local step at a timeupdate after every lossno cross-time backward

> **[Architecture Schematic]** *Variant 5 schematic with three detached local steps and immediate parameter updates θ0 to θ3.*
> *(Text in diagram: ⋯detachdetachstep 1h0fθh1ℓ1local bp + updatestep 2h1fθh2ℓ2local bp + updatestep 3h2fθh3ℓ3local bp + updateθ0θ1θ2θ3updateupdateupdate)*

The outer graph is still detached across time, but the runtime schedule changes: each local step is backpropagated and updated immediately, so θ0 → θ1 → θ2 → θ3 inside the rollout.

**What changes:** the detached local graph is the same as in Variant 4, but the training schedule changes.
This is the first row that changes the learning algorithm itself. Instead of waiting until the end of the rollout, every outer step now does forward  $\t o$  local backward  $\t o$  optimizer update immediately. Step  $t+1$  therefore sees parameters that have already been updated after step  $t$ .

From here on, the natural unit is **one optimizer interval = one outer step**. This is exactly the point where both the measurement unit and the learning algorithm change.

Per optimizer interval,

$$ C_{\tex t{shared, per-step, detach, instant}}^{\tex t{interval}}=c_{ℓ}+c_{B}+c_{u}. $$

and the peak activation memory is

$$ M_{\tex t{shared, per-step, detach, instant}}^{\tex t{peak}}=a_{f}+a_{ℓ}. $$

There is no  $c_{\thet a}$  term here, because there is no cross-step gradient accumulation before the optimizer update.

Over the same  $T$ -step horizon, the total cost is

$$ C_{\tex t{shared, per-step, detach, instant}}^{\tex t{rollout}}(T)=T(c_{ℓ}+c_{B}+c_{u}), $$

with peak activation memory

$$ M_{\tex t{shared, per-step, detach, instant}}^{\tex t{rollout}}=a_{f}+a_{ℓ}. $$

Within the numbered chain, this is the first row whose profiled optimizer interval is one outer step. That schedule has constant outer-depth activation peak. A detached streaming-backward implementation would already achieve the same outer-depth peak without changing the optimizer interval. What instant update adds on top is the optimizer-schedule change: the parameters are updated inside the rollout rather than after the rollout.

### V6. + internal truncation inside the cell

Variant 6 · instant update + internal truncation inside the cell
one local step at a timeupdate after every losstruncate inner loop

> **[Architecture Schematic]** *Variant 6 schematic with detached local steps, immediate updates, and a smaller purple inner truncated region inside each cell.*
> *(Text in diagram: ⋯detachdetachstep 1h0fθh1ℓ1local bp + updateinner trunc.step 2h1fθh2ℓ2local bp + updateinner trunc.step 3h2fθh3ℓ3local bp + updateinner trunc.θ0θ1θ2θ3updateupdateupdate)*

The outer one-step schedule is the same as Variant 5. The extra change is inside each cell: the purple dashed inner region marks a smaller cell-internal backward graph.

**What changes:** nothing at the outer schedule level. The only new truncation happens **inside** the cell-local backward.

Relative to Variant 5, the outer schedule is unchanged. The only new change is internal to the cell: if  $f_{\thet a}$  itself contains an inner loop, we can truncate gradients there as well.

This does not change the outer-step forward computation. It only reduces the local backward and local activation storage inside the cell.

Under the instant-update schedule of Variant 5, the exact cost per optimizer interval becomes

$$ C_{\tex t{shared, per-step, detach, instant, trunc}}^{\tex t{interval}}=c_{ℓ}+c_{B}^{\tex t{trunc}}+c_{u}, $$

where

$$ c_{B}^{\tex t{trunc}}\l e c_{B}, $$

typically with strict inequality in the intended truncated settings.

The peak activation memory per optimizer interval becomes

$$ M_{\tex t{shared, per-step, detach, instant, trunc}}^{\tex t{peak}}=a_{f}^{\tex t{trunc}}+a_{ℓ}, a_{f}^{\tex t{trunc}}\l e a_{f}, $$

again typically with strict inequality in the intended benchmark settings.

Over the same  $T$ -step horizon, the total compute is

$$ C_{\tex t{shared, per-step, detach, instant, trunc}}^{\tex t{rollout}}(T)=T(c_{ℓ}+c_{B}^{\tex t{trunc}}+c_{u}), $$

with peak activation memory still

$$ M_{\tex t{shared, per-step, detach, instant, trunc}}^{\tex t{rollout}}=a_{f}^{\tex t{trunc}}+a_{ℓ}. $$

This row combines two logically separate changes:

- instant update changes the optimizer schedule at the outer-loop level;
- internal truncation reduces the local backward and local activation memory inside each cell.

The forward cost of one outer step is unchanged; the savings come entirely from the local backward and activation-storage side of the cell.

This is the cleanest place in the literature to anchor HRM rather than TRM. Both use internal truncation, but with different truncation lengths.

- HRM backpropagates through only one inner step: HRM trains on repeated supervision segments, detaches the carried state between segments, updates the parameters after each segment, and uses a one-step gradient approximation for the final inner transition [[5]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
. That approximation is motivated as a practical surrogate for the Implicit Function Theorem-style gradient signal discussed in [Citation: Bai, *& al.* [6]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#bai2019deepequilibriummodels)
.
- TRM backpropagates through more inner steps: TRM keeps the same repeated-supervision, detach-between-segments, and per-segment update structure, but extends the gradient path across more inner steps [[1]](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#)
.

### V7. + gradient checkpointing

Variant 7 · instant update + internal truncation + gradient checkpointing
one local step at a timeupdate after every losstruncate inner loopcheckpoint remaining differentiable path

> **[Architecture Schematic]** *Variant 7 schematic with detached local steps, immediate updates, inner truncation, and checkpointing around the remaining differentiable part of each cell.*
> *(Text in diagram: ⋯detachdetachstep 1h0fθh1ℓ1local bp + updateinner trunc.grad ckptstep 2h1fθh2ℓ2local bp + updateinner trunc.grad ckptstep 3h2fθh3ℓ3local bp + updateinner trunc.grad ckptθ0θ1θ2θ3updateupdateupdate)*

This keeps the one-step truncated schedule of Variant 6, but now the remaining differentiable part of each cell is checkpointed: memory drops a bit further, and backward pays extra recomputation FLOPs.

**What changes:** keep the one-step outer schedule and the inner truncation, then checkpoint the remaining differentiable part of the cell.

Relative to Variant 6, the outer schedule is unchanged and the NFE is unchanged. The local gradient structure is still the truncated one-step version of Variant 6. The only new change is that the remaining differentiable part of the already truncated cell is now checkpointed instead of stored in the ordinary way.

So, per optimizer interval,

$$ C_{\tex t{shared, per-step, detach, instant, trunc, ckpt}}^{\tex t{interval}}=c_{ℓ}+c_{B}^{\tex t{trunc}}+c_{u}+r_{\tex t{ckpt}}^{\tex t{trunc}}. $$

The peak activation memory becomes

$$ M_{\tex t{shared, per-step, detach, instant, trunc, ckpt}}^{\tex t{peak}}=a_{f}^{\tex t{trunc,ckpt}}+a_{ℓ}, a_{f}^{\tex t{trunc,ckpt}}\l e a_{f}^{\tex t{trunc}}, $$

with strict inequality in the intended checkpointed settings.

Over the same  $T$ -step horizon,

$$ C_{\tex t{shared, per-step, detach, instant, trunc, ckpt}}^{\tex t{rollout}}(T)=T(c_{ℓ}+c_{B}^{\tex t{trunc}}+c_{u}+r_{\tex t{ckpt}}^{\tex t{trunc}}), $$

with peak activation memory still

$$ M_{\tex t{shared, per-step, detach, instant, trunc, ckpt}}^{\tex t{rollout}}=a_{f}^{\tex t{trunc,ckpt}}+a_{ℓ}. $$

The measured charts in the experiment section also include three **indented auxiliary rows** right after Variants 3 and 4: the fixed-schedule `3 + checkpointing`, `4 + streaming backward accumulation`, and `4 + checkpointing` comparisons, shown for apples-to-apples reference, but **not** counted as new numbered variants in the main ablation line.

#### ▾ Collapsible Section: How gradient checkpointing saves memory

Without checkpointing, autograd keeps the step-local activations produced inside each iterative cell so that backward can reuse them later. If we write

$$ a_{f}=a_{h}+a_{\tex t{int}}, $$

then  $a_{h}$  is the boundary hidden state and  $a_{\tex t{int}}$  are the internal saved tensors of that step.

With checkpointing, we keep only a smaller boundary representation during forward and drop most of the internal saved tensors. During backward, we rerun the forward of the checkpointed region to reconstruct the missing activations and only then apply the local backward. That is the whole tradeoff:

$$ a_{f}^{\tex t{ckpt}}\l e a_{f}, $$

with strict inequality in the intended checkpointed settings, and the checkpointed cell pays an extra  $r_{\tex t{ckpt}}$  recomputation FLOPs during backward.

Checkpointing saves memory by storing fewer forward activations and reconstructing them later. It does **not** change the gradient graph.

#### ▾ Collapsible Section: Shared vs non-shared under checkpointing

Checkpointing acts on activations, not on parameter tying. If two models have the same unrolled depth and the same per-step transition shape, then a fixed checkpointing policy saves the same kind of activation memory in both: it drops step-local internal activations and recomputes them during backward. Weight sharing does **not** make step- $t$  activations reusable for step  $t+1$ , because those activations come from different hidden states.

The differences appear elsewhere. Read “shared + checkpointing” as “non-shared + checkpointing,” plus the usual weight-sharing differences:

- parameter memory:  $T$  parameter blocks  $\t o$  one shared parameter block;
- gradient buffers:  $T$  separate parameter-gradient buffers  $\t o$  one shared gradient buffer;
- optimizer state:  $T$  optimizer-state blocks  $\t o$  one optimizer-state block;
- cross-step parameter accumulation: none  $\t o$  required.

So the clean summary is:

- checkpointing helps both shared and non-shared models because both store step-specific activations;
- weight sharing reduces parameter-side memory, not step-activation memory;
- checkpointing and weight sharing are complementary because they attack different parts of the footprint.

In practice, a shared-weight loop may re-enter the same block many times during checkpointed backward, while an untied model walks through different parameter blocks. The activation-memory story is the same; only the implementation overheads differ.

At this point, the symbolic story is complete:

- **shared weights:** mostly change parameter-side accumulation, not step-local activation storage;
- **per-step losses:** add local backward branches;
- **`detach`:** removes temporal credit assignment, not local backward;
- **instant update:** changes the optimizer interval from one full rollout to one outer step;
- **internal truncation:** shrinks the cell-local backward inside that new schedule;
- **checkpointing:** swaps saved activations for recomputation without changing the detached gradient estimator.

---

## Toy Experiment Check

To make the discussion less purely symbolic, I ran a small benchmark that mirrors the nested structure of the post. The exact benchmark script is [here](https://gist.github.com/huskydoge/a497e2412de02553c9b1d5b1742ce620).

- the **outer loop** rolls for  $T_{\tex t{out}}$  steps and owns the losses;
- each outer step applies one **iterative cell**;
- inside that cell, there is an **inner refinement loop** of depth  $K$ ;
- the internal-truncation variant differentiates only the **last** refinement, while the earlier refinements run without gradient tracking.

The tying convention in this toy benchmark is deliberately asymmetric. With the setting used below,  $T_{\tex t{out}}=32$  and  $K=6$ :

- **Variant 1 (non-shared / untied)** fully materializes the rollout into  $32\time s 6=192$  distinct refinement layers, so every refinement has its own parameter block;
- **Variants 2-7 (shared / tied)** use **one** outer-step cell shared across all  $32$  outer steps, with  $6$  inner refinements inside each call;
- the two setups therefore execute the same  $192$  logical refinement calls per full rollout, but Variant 1 realizes them as  $192$  separate parameter blocks, whereas Variants 2-7 reuse one shared block throughout.

Concretely, each outer step uses the same style of cell as the setup section,

$$ h_{t}=\tan h(W_{h}h_{t-1}+W_{x}x_{t}+b), $$

and in the batched benchmark implementation this means

$$ Z_{t}=H_{t-1}W_{h}^{⊤}+X_{t}W_{x}^{⊤}+1b^{⊤}, H_{t}=\tan h(Z_{t}), $$

with  $H_{t-1}\i n R^{B\time s D}$ ,  $X_{t}\i n R^{B\time s X}$ ,  $W_{h}\i n R^{D\time s D}$ ,  $W_{x}\i n R^{D\time s X}$ ,  $b\i n R^{D}$ , and  $1\i n R^{B}$  the all-ones vector. If  $A_{t}=dL/dH_{t}$  is the incoming batch adjoint, define

$$ S_{t}=1-H_{t}\odo t H_{t}, \ba r{A}_{t}=A_{t}\odo t S_{t}. $$

Then the two local backward objects specialized to this benchmark cell are

$$ A_{t}J_{t}=\ba r{A}_{t}W_{h}, $$

and

$$ A_{t}B_{t}=(\ba r{A}_{t}^{⊤}H_{t-1}, \ba r{A}_{t}^{⊤}X_{t}, 1^{⊤}\ba r{A}_{t}). $$

So, in the toy benchmark, the blue temporal term is the exported hidden-to-hidden multiply  $A_{t}J_{t}$ , while the red local term  $A_{t}B_{t}$  collects the gradients with respect to  $W_{h}$ ,  $W_{x}$ , and  $b$  inside one outer-step cell. The outer state then goes through one linear head and an MSE loss.

### Benchmark setup

The run below used one deliberately small but still nontrivial setting:

- batch size  $=32$
- outer rollout length  $T_{\tex t{out}}=32$
- hidden size  $d_{h}=256$
- input size  $d_{x}=256$
- inner cell depth  $K=6$
- Adam on CPU

For every row below, `torch.profiler` wraps the full optimizer-interval body: forward through the iterative cell and head, loss formation, backward, and `optimizer.step()`. So the FLOPs shown below are **profiler-estimated FLOPs per training interval**.

These profiler-estimated FLOPs should be read as a comparative operator-accounted proxy under one fixed implementation, not as a complete hardware-level accounting of all floating-point work.[^2]

Under the definition above, checkpointing does **not** change NFE; it only raises profiler-estimated FLOPs by recomputation during backward.

The two memory columns answer slightly different questions:

- **Peak saved activations:** how much backward-facing activation storage autograd kept alive.
- **Tracked peak memory:** peak saved activations plus model parameters, gradient buffers, and optimizer state inside this benchmark setup.

Because this run uses Adam, the optimizer-state breakdown is nontrivial rather than an all-zero auxiliary chart.

The table below maps the article variants to the benchmark implementation. Variant 4 uses detached carry with end-of-rollout backward, while an indented auxiliary row measures the same detached objective under streaming per-step backward accumulation with a delayed `optimizer.step()`.

| Row | Outer loss attachment | Carry between outer steps | Optimizer-step timing | Inner gradient scope | Checkpoint boundary |
| --- | --- | --- | --- | --- | --- |
| 1 baseline | terminal loss on the final outer state only | full graph carried across the rollout | once after full rollout | all 6 inner loops differentiated | none |
| 2 + shared | terminal loss on the final outer state only | full graph carried across the rollout | once after full rollout | all 6 inner loops differentiated | none |
| 3 + per-step losses | one outer loss at every step | full graph carried across the rollout | once after full rollout | all 6 inner loops differentiated | none |
| ↳ + checkpointing on top of 3 | one outer loss at every step | full graph carried across the rollout | once after full rollout | all 6 inner loops differentiated | checkpoint the full outer-step cell |
| 4 + `detach` each step under end-of-rollout backward | one outer loss at every step | value carried, graph detached between steps | once after full rollout | all 6 inner loops differentiated | none |
| ↳ + streaming backward accumulation on top of 4 | one outer loss at every step | value carried, graph detached between steps | backward after each outer step; optimizer step once after full rollout | all 6 inner loops differentiated | none |
| ↳ + checkpointing on top of 4 | one outer loss at every step | value carried, graph detached between steps | once after full rollout | all 6 inner loops differentiated | checkpoint the full outer-step cell |
| 5 + instant update | one outer loss on the profiled step | next step would start from detached carried state | once per outer step | all 6 inner loops differentiated | none |
| 6 + internal truncation | one outer loss on the profiled step | next step would start from detached carried state | once per outer step | only the last inner loop is differentiable | none |
| 7 + gradient checkpointing | one outer loss on the profiled step | next step would start from detached carried state | once per outer step | only the last inner loop is differentiable | checkpoint only that remaining differentiable inner loop |

The measurement contract changes across the chain, so the empirical results should be read in two blocks: Rows 1-4 plus the three auxiliary fixed-schedule rows are measured per **full 32-step outer rollout**, whereas Rows 5-7 are measured per **single outer step**. When I want apples-to-apples compute across schedules, I normalize Rows 5-7 back to the same 32-step outer horizon.

### Results

Read `↓` as the main numbered ablation chain and `↳` as an auxiliary branch on the immediately preceding numbered variant. Rows 1-4 plus the three indented auxiliary rows are full-rollout intervals, whereas Rows 5-7 are one-step intervals, so the figure should be read as two blocks rather than one monotone scale. FLOPs are `torch.profiler(with_flops=True)` estimates, and NFE means the logical number of forward cell refinements in the modeled computation, so checkpointing can raise FLOPs without changing NFE; keep the aggregate peak-memory chart visible first and expand the breakdown only when you want the components.

**Chart: NFE**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 192.00 |
| ↓ 2 +shared | 192.00 |
| ↓ 3 +losses | 192.00 |
| ↳ +gradckpt (3) | 192.00 |
| ↓ 4 +detach | 192.00 |
| ↳ +stream bwd (4) | 192.00 |
| ↳ +gradckpt (4) | 192.00 |
| ↓ 5 +instant | 6.00 |
| ↓ 6 +trunc | 6.00 |
| ↓ 7 +gradckpt | 6.00 |

**Chart: Profiler-estimated FLOPs**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 4038.07 |
| ↓ 2 +shared | 4038.20 |
| ↓ 3 +losses | 4428.33 |
| ↳ +gradckpt (3) | 6042.09 |
| ↓ 4 +detach | 4298.31 |
| ↳ +stream bwd (4) | 4302.32 |
| ↳ +gradckpt (4) | 5912.07 |
| ↓ 5 +instant | 134.45 |
| ↓ 6 +trunc | 71.40 |
| ↓ 7 +gradckpt | 121.83 |

**Chart: Peak Memory**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 451.85 |
| ↓ 2 +shared | 69.10 |
| ↓ 3 +losses | 79.76 |
| ↳ +gradckpt (3) | 16.01 |
| ↓ 4 +detach | 72.01 |
| ↳ +stream bwd (4) | 5.16 |
| ↳ +gradckpt (4) | 16.01 |
| ↓ 5 +instant | 5.16 |
| ↓ 6 +trunc | 3.45 |
| ↓ 7 +gradckpt | 3.41 |

#### ▾ Collapsible Section: Expand peak-memory breakdown (activations / parameters / gradients / optimizer state)

**Chart: Saved Activations**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 66.09 |
| ↓ 2 +shared | 66.09 |
| ↓ 3 +losses | 76.75 |
| ↳ +gradckpt (3) | 13.00 |
| ↓ 4 +detach | 69.00 |
| ↳ +stream bwd (4) | 2.16 |
| ↳ +gradckpt (4) | 13.00 |
| ↓ 5 +instant | 2.16 |
| ↓ 6 +trunc | 0.44 |
| ↓ 7 +gradckpt | 0.41 |

**Chart: Peak Model Parameter Memory**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 96.44 |
| ↓ 2 +shared | 0.75 |
| ↓ 3 +losses | 0.75 |
| ↳ +gradckpt (3) | 0.75 |
| ↓ 4 +detach | 0.75 |
| ↳ +stream bwd (4) | 0.75 |
| ↳ +gradckpt (4) | 0.75 |
| ↓ 5 +instant | 0.75 |
| ↓ 6 +trunc | 0.75 |
| ↓ 7 +gradckpt | 0.75 |

**Chart: Gradient Buffers**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 96.44 |
| ↓ 2 +shared | 0.75 |
| ↓ 3 +losses | 0.75 |
| ↳ +gradckpt (3) | 0.75 |
| ↓ 4 +detach | 0.75 |
| ↳ +stream bwd (4) | 0.75 |
| ↳ +gradckpt (4) | 0.75 |
| ↓ 5 +instant | 0.75 |
| ↓ 6 +trunc | 0.75 |
| ↓ 7 +gradckpt | 0.75 |

**Chart: Optimizer State**

| Variant / Stage | Value |
| --- | --- |
| 1 baseline | 192.88 |
| ↓ 2 +shared | 1.50 |
| ↓ 3 +losses | 1.50 |
| ↳ +gradckpt (3) | 1.50 |
| ↓ 4 +detach | 1.50 |
| ↳ +stream bwd (4) | 1.50 |
| ↳ +gradckpt (4) | 1.50 |
| ↓ 5 +instant | 1.50 |
| ↓ 6 +trunc | 1.50 |
| ↓ 7 +gradckpt | 1.50 |

The charts keep all rows in one compact visual stack. Read `↓` in the left labels as the main numbered ablation chain and `↳` as an auxiliary branch that modifies only the immediately preceding numbered variant.

#### ▾ Collapsible Section: Why V3 and V4 are close? A detailed analysis

Start from the two formulas already derived above:

$$ C_{\tex t{shared, per-step}}(T)=Tc_{ℓ}+(T-1)c_{J}+(T-1)c_{h}+Tc_{B}+(T-1)c_{\thet a}+c_{u}, $$

and

$$ C_{\tex t{shared, per-step, detach}}(T)=Tc_{ℓ}+Tc_{B}+(T-1)c_{\thet a}+c_{u}. $$

So the entire V3  $\t o$  V4 gap is

$$ C_{\tex t{shared, per-step}}(T)-C_{\tex t{shared, per-step, detach}}(T)=(T-1)c_{J}+(T-1)c_{h}. $$

That is the right entry point. The only real question is how large  $c_{J}+c_{h}$  is, in this toy benchmark, relative to the local term  $c_{ℓ}+c_{B}+c_{\thet a}$ .

For the benchmark cell used here, with batch size  $B$ , hidden size  $D$ , input size  $X$ , and inner depth  $K$ , a batched matrix multiply of shape  $[B,D]\time s [D,D]$  costs  $2BD^{2}$  FLOPs, and one of shape  $[B,X]\time s [X,D]$  costs  $2BXD$ .

The benchmark uses a square linear head on the outer state. If  $H_{t}\i n R^{B\time s D}$  is the batch of outer states at step  $t$ ,  $U\i n R^{D\time s D}$  is the head matrix, and  $Y_{t}\i n R^{B\time s D}$  is the target, then

$$ O_{t}=H_{t}U, ℓ_{t}=\fra c{1}{2}‖O_{t}-Y_{t}‖_{F}^{2}. $$

Writing  $\Delt a_{t}=O_{t}-Y_{t}$ , the head backward is

$$ \fra c{∂ℓ_{t}}{∂H_{t}}=\Delt a_{t}U^{⊤}, \fra c{∂ℓ_{t}}{∂U}=H_{t}^{⊤}\Delt a_{t}. $$

Each term is one  $[B,D]\time s [D,D]$  GEMM, so each costs  $2BD^{2}$ . Therefore

$$ c_{ℓ}=4BD^{2}. $$

Likewise, in the earlier symbolic notation  $c_{J}=e_{J}+m_{J}$ . For this affine-tanh cell,  $e_{J}$  is the elementwise derivative preparation

$$ S_{t}=1-H_{t}\odo t H_{t}, \ba r{A}_{t}=A_{t}\odo t S_{t}, $$

while  $m_{J}$  is the exported temporal VJP

$$ \ba r{A}_{t}W_{h}, $$

which is one more  $[B,D]\time s [D,D]$  GEMM. So

$$ m_{J}=2BD^{2}. $$

The remaining  $e_{J}$  work is only elementwise  $O(BD)$  and is not stably covered by `torch.profiler(with_flops=True)` in this benchmark. Under the profiler-aligned leading-term accounting used in this collapse, I therefore keep only the dominant counted GEMM part and write

$$ c_{J}\appro x m_{J}=2BD^{2}. $$

$$ c_{h}\appro x 0, c_{\thet a}\appro x 0, c_{u}\appro x 0, $$

because these are elementwise/add/update terms and `torch.profiler(with_flops=True)` does not give stable FLOPs coverage for them in this benchmark;

and, under the same profiler-aligned leading-term accounting,

$$ c_{B}\appro x K(2BD^{2}+2BXD)+(K-1)2BD^{2}, $$

because the local parameter backward still traverses all  $K$  inner refinements. At each refinement, exposing the parameter-gradient contribution costs one hidden-hidden term plus one input-hidden term, namely  $2BD^{2}+2BXD$ . In addition, to reach the earlier parameter uses inside the same cell, backward must still propagate the hidden adjoint across the first  $K-1$  inner links, each costing another  $2BD^{2}$ .

With the benchmark setting

$$ T=32, B=32, D=X=256, K=6, $$

this becomes

$$ c_{ℓ}=8.389\tex t{M}, c_{J}\appro x 4.194\tex t{M}, c_{B}\appro x 71.303\tex t{M}. $$

So the formulas predict

$$ (T-1)c_{J}+(T-1)c_{h}\appro x 31⋅4.194\tex t{M}=130.023\tex t{M}, $$

while the local part retained by **both** Variant 3 and Variant 4 is

$$ T(c_{ℓ}+c_{B})+(T-1)c_{\thet a}+c_{u}\appro x 32(8.389+71.303)\tex t{M}=2550.137\tex t{M}. $$

That is why the relative compute drop is modest: `detach` removes the temporal term, but the local loss branch and the local parameter-backward term are still the dominant pieces in this toy.

Up to this point, I have only compared the backward-side terms that differ between the two rows. To compare against the profiler totals, we now add back the common forward work that the symbolic model deliberately held fixed. For both Variant 3 and Variant 4, each of the  $T$  outer steps runs  $K$  cell refinements and one head forward, so the shared forward baseline is

$$ T(K(2BD^{2}+2BXD)+2BD^{2})=1744.830\tex t{M}, $$

where  $K(2BD^{2}+2BXD)$  is the cell forward and the final  $2BD^{2}$  is the head forward on that outer step. Therefore the full analytic interval totals are

$$ \underse t{\tex t{shared forward baseline}}{\underbrac e{1744.830}}+\underse t{\tex t{Variant 3 backward-side terms}}{\underbrac e{2680.161}}=4424.991\tex t{M for Variant 3}, $$

and

$$ \underse t{\tex t{shared forward baseline}}{\underbrac e{1744.830}}+\underse t{\tex t{Variant 4 backward-side terms}}{\underbrac e{2550.137}}=4294.967\tex t{M for Variant 4}, $$

which line up with the measured profiler-estimated values

$$ 4428.334\tex t{M}, 4298.310\tex t{M}. $$

So the small V3  $\t o$  V4 gap is not evidence of an implementation bug. It is exactly what the earlier formulas predict once the toy cell is instantiated under the same decomposition.

In real systems, the visual separation can be even less clean: parts of the local derivative preparation for the temporal VJP and the parameter VJP may be shared or fused inside the same backward kernels. That is why the split into  $c_{J}$  and  $c_{B}$  should still be read as a convenient decomposition, not as two physically separate kernel launches.

### Key Takeaways

The benchmark tracks the symbolic story closely: the biggest shifts come from changing which backward path exists and when local graphs are released, not from weight sharing alone.

- **Weight sharing collapses parameter-side memory, not activation-side memory.** Variant 1 and Variant 2 keep the same saved-activation peak at 66.094 MB, but tracked peak memory drops from 451.850 MB to 69.102 MB because the benchmark moves from a fully materialized 192-layer untied stack to one shared recurrent block, collapsing parameters, gradient buffers, and optimizer state.
- **Per-step losses make the local term dominant.** Variant 2  $\t o$  Variant 3 raises profiler-estimated FLOPs from 4038.198 M to 4428.334 M, and Variant 3  $\t o$  Variant 4 then falls only to 4298.310 M because `detach` removes only the temporal part while the local loss and local parameter-backward work remain; even after that drop, saved activations are still 69.000 MB under end-of-rollout backward.
- **Once per-step detach is introduced, streaming backward should be the default implementation.** After detaching at every step, the local graphs are already temporally disconnected, so backpropagating only at the end of the rollout brings essentially no computational benefit while needlessly retaining activations. In our benchmark, the auxiliary streaming row keeps essentially the same compute as Variant 4, 4302.316 M versus 4298.310 M, but reduces saved activations from 69.000 MB to 2.156 MB by releasing each detached local graph immediately.

## Future Topics

This post isolates the smallest ablation chain that makes gradient paths, optimizer intervals, and storage policies explicit. The next natural extensions are adaptive or routed recurrence, such as halting or mixture-of-recursions, where NFE becomes input dependent and the accounting has to move from a fixed  $T$  to expected depth. A third direction is systems work, especially FSDP and compiler interactions, where communication, sharding boundaries, and recomputation matter just as much as local FLOPs and saved activations.

### References

[1]**Less is More: Recursive Reasoning with Tiny Networks**A. Jolicoeur-Martineau, (2025)[Link](https://arxiv.org/abs/2510.04871)[2]**Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. Bartoldson, B. Kailkhura, A. Bhatele, T. Goldstein, (2025)[Link](https://arxiv.org/abs/2502.05171)[3]**Parcae: Scaling Laws For Stable Looped Language Models**H. Prairie, Z. Novack, T. Berg-Kirkpatrick, D. Fu, (2026)[Link](https://arxiv.org/abs/2604.12946)[4]**Scaling latent reasoning via looped language models**R. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, H. Que, B. Wei, Z. Wen, F. Yin, H. Xing, others, (2025)[5]**Hierarchical Reasoning Model**G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, Y. Yadkori, (2025)[Link](https://arxiv.org/abs/2506.21734)[6]**Deep Equilibrium Models**S. Bai, J. Kolter, V. Koltun, (2019)[Link](https://arxiv.org/abs/1909.01377)

---

1. In reverse-mode autodiff, backward through a step usually needs some forward-time tensors again, such as the input hidden state, preactivation, normalization statistics, or attention masks. Unless we choose a recomputation policy such as checkpointing, the framework therefore saves those tensors during forward so the later backward pass can form the local derivatives. [↩︎](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#fnref:1)
2. PyTorch documents `with_flops=True` as using formulas to estimate FLOPs of specific operators, so I treat the resulting number as a comparative operator-accounted proxy rather than a full hardware-level total. [https://docs.pytorch.org/docs/stable/profiler.html](https://docs.pytorch.org/docs/stable/profiler.html) [↩︎](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/#fnref:2)

Cited as

Use the plain citation or copy the BibTeX entry below.

> Benhao Huang. (Apr 2026). Loop-Model FLOPs and Memory in an Ablation Chain.
> Husky's Log.
> [/husky-blog/posts/recursive_models/loop-cost/](https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/)

BibTeXCopy

```bibtex
@article{huang2026loopmodel,
  title   = "Loop-Model FLOPs and Memory in an Ablation Chain",
  author  = "Benhao Huang",
  journal = "Husky's Log",
  year    = "2026",
  month   = "Apr",
  url     = "https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/"
}
```

- [Recursive Models](https://huskydoge.github.io/husky-blog/tags/recursive-models/)
- [BPTT](https://huskydoge.github.io/husky-blog/tags/bptt/)
- [Optimization](https://huskydoge.github.io/husky-blog/tags/optimization/)
- [Credit Assignment](https://huskydoge.github.io/husky-blog/tags/credit-assignment/)

[« PrevExact Input Writes Improve Stable Looped Language Models](https://huskydoge.github.io/husky-blog/posts/recursive_models/improve-parcae/)[Next »Paper Reading | Reasoning with Power Sampling](https://huskydoge.github.io/husky-blog/posts/paper-reading/reasoning-with-sampling/)

- 
- 
- 
- 
- 
- 
- 

Expanded Table-
Fit
100%
+
100%CloseExpanded FigureClose

### Reference: Cost Legend

- $c_{ℓ}$ : one local loss backward
- $e_{J},m_{J}$ : split temporal-Jacobian costs; together  $c_{J}=e_{J}+m_{J}$
- $e_{B},m_{B}$ : split local parameter-backward costs; together  $c_{B}=e_{B}+m_{B}$
- $c_{h}$ : one hidden-state gradient addition
- $c_{\thet a}$ : one shared-gradient accumulation
- $c_{u}$ : one optimizer update for one parameter block
- $r_{\tex t{ckpt}}$ : extra recomputation FLOPs from checkpointing
- $c_{B}^{\tex t{trunc}}$ : local parameter-backward cost under internal cell truncation
- $r_{\tex t{ckpt}}^{\tex t{trunc}}$ : extra recomputation FLOPs for checkpointing the remaining differentiable part of a truncated cell
- $a_{f}$ : activation memory of one iterative cell
- $a_{ℓ}$ : activation memory of one loss/head branch
- $a_{f}^{\tex t{ckpt}}$ : activation memory of one checkpointed iterative cell
- $a_{f}^{\tex t{trunc}}$ : activation memory of one iterative cell under internal cell truncation
- $a_{f}^{\tex t{trunc,ckpt}}$ : activation memory of one truncated iterative cell under checkpointing

### Sidenotes / Footnotes

[^1]: In reverse-mode autodiff, backward through a step usually needs some forward-time tensors again, such as the input hidden state, preactivation, normalization statistics, or attention masks. Unless we choose a recomputation policy such as checkpointing, the framework therefore saves those tensors during forward so the later backward pass can form the local derivatives.

[^2]: PyTorch documents with_flops=True as using formulas to estimate FLOPs of specific operators, so I treat the resulting number as a comparative operator-accounted proxy rather than a full hardware-level total. https://docs.pytorch.org/docs/stable/profiler.html