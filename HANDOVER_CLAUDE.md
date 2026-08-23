# Handover Briefing for Claude (Opus)

**Timestamp:** 2026-08-23 16:52 MSK  
**Prepared by:** Gemini (Antigravity Agent)  
**Submission Deadline:** 2026-08-23 23:59 MSK (~7 hours remaining)

---

## 1. Executive Status & Operational Rule

Per user instruction, **no further speculative code edits or new jobs will be launched**. All active local and remote runs are isolated, progressing smoothly, and scheduled to complete between **17:04 and 17:45 MSK**.

All research questions raised in prior turns have been tested with concrete code, empirical measurements, or active training arms.

---

## 2. Live Compute Matrix & Harvest Protocol

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       LIVE RUNNING JOBS                                                │
├───────────────────┬──────────────────────────┬───────────────────────┬────────────┬────────────────────┤
│ Platform / Node   │ Job / Task Identifier    │ Arm / Configuration   │ Status     │ Target Finish Time │
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **Apple MPS**     │ task-341                 │ od_lora_r2 (Arm 2)    │ Step 207+  │ ~17:04 MSK         │
│ (Local Sequential)│ (run_operator_diversity) │ od_lora_r4 (Arm 3)    │ Queued     │ ~17:24 MSK         │
│                   │                          │ od_depth_gate (Arm 4) │ Queued     │ ~17:44 MSK         │
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **DataSphere T4** │ bt1sglqurmj6frrmsfrk     │ Paired 3-Arm Benchmark│ EXECUTING  │ ~17:10 MSK         │
│ (Cloud Slot 1)    │ (tlab-operator-diversity)│ (control, lora, gate) │            │                    │
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **DataSphere T4** │ bt1vqefjccioapof5fgh     │ Deep Full-Budget      │ Step 14,150│ Switches to k=1 @  │
│ (Cloud Slot 2)    │ (tlab-deep-full)         │ (mu_rec=40, sw75)     │ / 19,531   │ 14,648 (~16:55 MSK)│
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **DataSphere T4** │ bt1hp97su48dc6096sqn     │ sw90 Anchor Falsifier │ EXECUTING  │ ~17:45 MSK         │
│ (Cloud Slot 3)    │ (tlab-anchor-tokenkey)   │ (k in {2, 3, 5})      │            │                    │
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **Kaggle T4 #1**  │ tlab-od-rank8-sweep      │ LoRA r=8 (Seed 1)     │ RUNNING    │ ~17:08 MSK         │
├───────────────────┼──────────────────────────┼───────────────────────┼────────────┼────────────────────┤
│ **Kaggle T4 #2**  │ tlab-od-annealed-sw90    │ sw90 + LoRA r=4       │ RUNNING    │ ~17:08 MSK         │
└───────────────────┴──────────────────────────┴────────────────────────┴────────────┴────────────────────┘
```

### Exact CLI Commands for Harvesting:
1. **Local MPS Sweep:**
   * Log: `tail -n 30 .system_generated/tasks/task-341.log`
   * Output JSON: [`checkpoints/operator_diversity_results.json`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/checkpoints/operator_diversity_results.json)
2. **DataSphere `bt1sglqurmj6frrmsfrk`:**
   ```bash
   export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"
   GRPC_DNS_RESOLVER=native datasphere project job download-files --id bt1sglqurmj6frrmsfrk --with-logs --output-dir /tmp/harvest_ds_od
   ```
3. **DataSphere `bt1vqefjccioapof5fgh` (`tlab-deep-full`):**
   ```bash
   GRPC_DNS_RESOLVER=native datasphere project job download-files --id bt1vqefjccioapof5fgh --with-logs --output-dir /tmp/harvest_deep_full
   ```
4. **Kaggle Kernels:**
   ```bash
   kaggle kernels output arsen4ikvar/tlab-od-rank8-sweep -p /tmp/harvest_kg_rank8
   kaggle kernels output arsen4ikvar/tlab-od-annealed-sw90 -p /tmp/harvest_kg_sw90
   ```

---

## 3. Verified Empirical Results Completed in this Session

### A. Local Baseline Arm 1 (`od_control` Completed @ 16:44 MSK)
* **Final Validation CE Curve (2.5M tokens):**
  $$r=1: 5.4375 \to r=4: 5.3559 \to r=8: 5.3395 \to r=12: \mathbf{5.3391} \to r=16: 5.3435 \to r=24: 5.3571 \to r=32: 5.3729$$
* **Baseline Loop Gain:** **$0.0984\text{ nats}$** ($5.4375 \to 5.3391$ at loop 12).

### B. Experiment E3: Token Depth Conflict Probe ([`checkpoints/token_conflict_results.json`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/checkpoints/token_conflict_results.json))
* **Evaluated on 32,768 tokens ($N=128\text{ seq}$):**
  * Shallow-Oracle Tokens ($d^* \le 8, 51.7\%$ of tokens): Norm inflated **$3.41\times$** at depth 20; Cosine fidelity $\cos(h_{20}, h_{d^*})$ dropped to **$0.9419$** ($\Delta \cos \approx 0.0581$).
  * Deep-Oracle Tokens ($d^* \ge 16, 39.3\%$ of tokens): Norm ratio **$0.76\times$**; Cosine fidelity **$0.9983$**.
  * **Mechanistic Proof:** Un-gated deep execution actively corrupts the representations of over half the tokens in the sequence.

### C. Experiment B: Oracle Ragged KV Cache Probe ([`checkpoints/oracle_cache_results.json`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/checkpoints/oracle_cache_results.json))
* **KV Invariance:** Substituting past-token KV caches computed at shallow depths ($k \in \{1, 2, 4\}$) shifts cross-entropy by at most $\mathbf{0.0004\text{ nats}}$ at depth 20.
* **Ragged Oracle Gain:** At query depth 8, ragged oracle caching achieves $\mathbf{\text{CE} = 5.5104}$ ($\mathbf{-0.0079\text{ nats}}$ lower than uniform cache).

### D. Step-0 Mechanistic Collapse Pre-Test
* Scalar conditioning (`cond_scalar`) fails ($\min \cos \approx 0.9985$) because RMSNorm is scale-invariant.
* LoRA branch cycling (`cond_mode="lora_cycle"`, $r=4$) shatters collapse, dropping $\min \cos$ to **$0.6079$**.

---

## 4. Code & Architecture Invariants

1. **`LoRALayerAdapter` in [`src/model.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/model.py):**
   * $B$ projection matrices are explicitly zero-initialized ($B=0$) in `LoopedTransformer.__init__`. Step-0 bit-identity is verified ($\max|\text{diff}| = \mathbf{0.00\text{e}+00}$).
   * Parameter cost: $O(N \cdot r \cdot d)$. Linear in hidden dimension $d$, zero routing overhead.
   * LoRA $r=2$ (4 branches): $9,268,896\text{ params}$ ($+2.25\% \le 10\text{M}$).
   * LoRA $r=4$ (4 branches): $9,473,184\text{ params}$ ($+4.03\% \le 10\text{M}$).
2. **State Depth Gate in [`src/model.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/model.py):**
   * `self.depth_gate_head = nn.Linear(H, 1, bias=False)` ($+448\text{ params}$), zero-initialized.
   * State-conditioned convex mixture $\alpha_t(h_t) = \text{Softmax}(w^T h_t)$ before readout.
3. **Commit SHAs (Gemini Attribution):**
   * `d49e119`: Architecture, analytical budget accounting, and test suite.
   * `6a6130d`: Paired 4-arm local benchmark script `src/run_operator_diversity.py`.

---

## 5. File Inventory

* [`PROGRESS_REPORT.md`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/PROGRESS_REPORT.md): Exhaustive synchronized research and operational ledger.
* [`src/model.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/model.py)
* [`src/param_budget.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/param_budget.py)
* [`src/test_model.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/test_model.py)
* [`src/run_operator_diversity.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/run_operator_diversity.py)
* [`src/oracle_cache_probe.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/oracle_cache_probe.py)
* [`src/token_depth_conflict_probe.py`](file:///Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer/src/token_depth_conflict_probe.py)
* Scratchpad directories:
  * `/private/tmp/claude-501/.../scratchpad/ds_operatordiversity/`
  * `/private/tmp/claude-501/.../scratchpad/kg_od_rank8/`
  * `/private/tmp/claude-501/.../scratchpad/kg_od_sw90/`
