# All experiments — the complete inventory

*The task asks for «подробное описание всех экспериментов». This is that, at arm level: every training
run this project produced a validation curve for, with its configuration, budget and result. It is
**generated from the stored artifacts**, not transcribed, so it cannot drift from them —
regenerate with `python src/make_inventory.py`.*

**How to read it.** One row = one trained model. `band`/`mid` are the useful-depth plateau
(`src/plateau.py`, tolerance 0.01) and its geometric midpoint; **`mid = √(onset × end)`, so it is one
number standing for two independent quantities** — how far the model keeps *improving* (onset) and how
long before degradation exceeds tolerance (end). Compare `mid` only within a shared `grid`: grid
choice alone moves it 17%.

**Comparisons are only meaningful within a source file.** Each `*_results.json` is one job, so arms
sharing a source share a data shard, tokenizer and seed and are *in-job paired*. Across sources this
project measures 0.0074–0.0334 nats of drift, which is larger than several effects claimed here.

## Coverage — checked mechanically, and it is not perfect

Of the 128 arms, **71 appear in `report.md` by their internal identifier and 111 by their best-CE value; 121 by one or the other.** **7 appear by neither:**

| arm | best CE | why it is absent, and whether that is a defect |
|---|---|---|
| `as_10M_sw90` | 4.3938 | §4.17 reports its **ΔCE_best = −0.0764** against the in-job control, not its absolute |
| `sc_final_only_s1` | 5.0624 | §4.6b reports its **Δ** (−0.5165), not its absolute |
| `trainL16_s1` | 4.6202 | §4.9's seed-1 replication reports the **re-zeroed mean curve** and the per-arm shape spread, never the five absolute bests. The conclusion (the collapse fails its pre-registered seed test) is reported; the raw column is not |
| `trainL2_s1` | 4.4566 | §4.9's seed-1 replication reports the **re-zeroed mean curve** and the per-arm shape spread, never the five absolute bests. The conclusion (the collapse fails its pre-registered seed test) is reported; the raw column is not |
| `trainL32_s1` | 4.6055 | §4.9's seed-1 replication reports the **re-zeroed mean curve** and the per-arm shape spread, never the five absolute bests. The conclusion (the collapse fails its pre-registered seed test) is reported; the raw column is not |
| `trainL4_s1` | 4.4772 | §4.9's seed-1 replication reports the **re-zeroed mean curve** and the per-arm shape spread, never the five absolute bests. The conclusion (the collapse fails its pre-registered seed test) is reported; the raw column is not |
| `trainL8_s1` | 4.5601 | §4.9's seed-1 replication reports the **re-zeroed mean curve** and the per-arm shape spread, never the five absolute bests. The conclusion (the collapse fails its pre-registered seed test) is reported; the raw column is not |

**So there is no unreported *experiment*** — every arm above is a run whose conclusion is reported; what is absent is an *absolute number* where the report quotes a delta or a mean curve instead. This table is where those numbers live.

## The inventory

Total arms with a final validation curve: **128**

| # | arm | source | dev | tokens | loops | k | CE@1 | best CE | @r | band | mid | grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `local_anneal_sw75_s0` | anneal_local_results | mps | 2.50M | 4-32 | 5 | 5.5767 | **5.4013** | 16 | [12,24] | 17.0 | 8 |
| 2 | `a3_dense_s0` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.4463 | **5.3391** | 8 | [8,16] | 11.3 | 11 |
| 3 | `a3_dense_s1` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.4675 | **5.3642** | 8 | [8,16] | 11.3 | 11 |
| 4 | `a3_sw75_s0` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.4647 | **5.2735** | 16 | [12,24] | 17.0 | 11 |
| 5 | `a3_sw75_s1` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.7303 | **5.4548** | 20 | [12,24] | 17.0 | 11 |
| 6 | `a3_sw90_s0` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.4047 | **5.2580** | 12 | [8,24] | 13.9 | 11 |
| 7 | `a3_sw90_s1` | anneal_rep2_results | cuda | 2.50M | 4-32 | 5 | 5.4397 | **5.3033** | 16 | [8,24] | 13.9 | 11 |
| 8 | `a2_an50_s0` | anneal_rep_results | cuda | 2.50M | 4-32 | 5 | 5.5839 | **5.3443** | 20 | [12,24] | 17.0 | 11 |
| 9 | `a2_an50_s1` | anneal_rep_results | cuda | 2.50M | 4-32 | 5 | 5.6930 | **5.4604** | 20 | [12,32] | 19.6 | 11 |
| 10 | `a2_dense_s0` | anneal_rep_results | cuda | 2.50M | 4-32 | 5 | 5.4410 | **5.3418** | 8 | [8,16] | 11.3 | 11 |
| 11 | `a2_dense_s1` | anneal_rep_results | cuda | 2.50M | 4-32 | 5 | 5.4843 | **5.3816** | 8 | [8,16] | 11.3 | 11 |
| 12 | `an_rev50` | anneal_results | cuda | 2.50M | 4-32 | 1 | 5.6914 | **5.5957** | 12 | [8,16] | 11.3 | 11 |
| 13 | `an_sw50` | anneal_results | cuda | 2.50M | 4-32 | 5 | 5.6078 | **5.3711** | 20 | [12,24] | 17.0 | 11 |
| 14 | `an_sw75` | anneal_results | cuda | 2.50M | 4-32 | 5 | 5.4891 | **5.3061** | 16 | [12,24] | 17.0 | 11 |
| 15 | `an_sw90` | anneal_results | cuda | 2.50M | 4-32 | 5 | 5.4154 | **5.2659** | 16 | [8,24] | 13.9 | 11 |
| 16 | `as_10M_dense` | anneal_scale_10M_results | cuda | 10.00M | 4-32 | 5 | 4.6497 | **4.4702** | 12 | [8,16] | 11.3 | 11 |
| 17 | `as_10M_sw90` | anneal_scale_10M_results | cuda | 10.00M | 4-32 | 5 | 4.6405 | **4.3938** | 12 | [8,20] | 12.6 | 11 |
| 18 | `baseline_cuda_lr0.0005` | baseline_cuda_results | cuda | 6.00M | 1-1 | 1 | 4.4422 | **4.4422** | 1 | [1,1] | 1.0 | 1 |
| 19 | `baseline_cuda_lr0.001` | baseline_cuda_results | cuda | 6.00M | 1-1 | 1 | 4.3742 | **4.3742** | 1 | [1,1] | 1.0 | 1 |
| 20 | `baseline_cuda_lr0.003` | baseline_cuda_results | cuda | 6.00M | 1-1 | 1 | 4.4651 | **4.4651** | 1 | [1,1] | 1.0 | 1 |
| 21 | `gate_control` | convex_gate_results | cuda | 10.00M | 4-32 | 5 | 4.6426 | **4.4518** | 8 | [8,16] | 11.3 | 11 |
| 22 | `gate_convex` | convex_gate_results | cuda | 10.00M | 4-32 | 5 | 4.6549 | **4.4722** | 8 | [8,16] | 11.3 | 11 |
| 23 | `null_rep1` | cuda_null_results | cuda | 2.50M | 4-32 | 5 | 5.4543 | **5.3392** | 12 | [8,16] | 11.3 | 11 |
| 24 | `null_rep2` | cuda_null_results | cuda | 2.50M | 4-32 | 5 | 5.4355 | **5.3304** | 8 | [8,16] | 11.3 | 11 |
| 25 | `null_rep3` | cuda_null_results | cuda | 2.50M | 4-32 | 5 | 5.4358 | **5.3242** | 12 | [8,16] | 11.3 | 11 |
| 26 | `da_mu40_dense` | deep_anneal_mu40_results | cuda | 2.50M | 32-48 | 5 | 5.6513 | **5.4658** | 24 | [16,40] | 25.3 | 12 |
| 27 | `da_mu40_sw75` | deep_anneal_mu40_results | cuda | 2.50M | 32-48 | 5 | 5.8263 | **5.4466** | 48 | [32,64] | 45.3 | 12 |
| 28 | `da_mu40_sw90` | deep_anneal_mu40_results | cuda | 2.50M | 32-48 | 5 | 5.7262 | **5.4394** | 32 | [24,48] | 33.9 | 12 |
| 29 | `df_mu40_sw75` | deep_full_results | cuda | 40.00M | 32-48 | 5 | 4.4864 | **3.9287** | 24 | [16,32] | 22.6 | 14 |
| 30 | `d3_mu40_dense` | deep_mu40_results | cuda | 2.50M | 32-48 | 5 | 5.6122 | **5.4170** | 24 | [16,32] | 22.6 | 11 |
| 31 | `d3_mu40_term` | deep_mu40_results | cuda | 2.50M | 32-48 | 1 | 6.6313 | **5.6051** | 48 | [32,48] | 39.2 | 11 |
| 32 | `dt_mu18_dense` | deep_terminal_results | cuda | 2.50M | 4-32 | 5 | 5.4381 | **5.3329** | 8 | [8,16] | 11.3 | 11 |
| 33 | `dt_mu18_term` | deep_terminal_results | cuda | 2.50M | 4-32 | 1 | 5.7844 | **5.5242** | 16 | [16,24] | 19.6 | 11 |
| 34 | `dt_mu32_dense` | deep_terminal_results | cuda | 2.50M | 24-40 | 5 | 5.5880 | **5.4148** | 16 | [16,32] | 22.6 | 11 |
| 35 | `dt_mu32_term` | deep_terminal_results | cuda | 2.50M | 24-40 | 1 | 6.3054 | **5.4850** | 32 | [32,32] | 32.0 | 11 |
| 36 | `dc_control_s0` | ds_dc0 | cuda | 3.50M | 4-32 | 5 | 5.2631 | **5.1577** | 12 | [8,20] | 12.6 | 11 |
| 37 | `dc_w2_s0` | ds_dc0 | cuda | 3.50M | 4-32 | 5 | 5.2857 | **5.1670** | 12 | [8,20] | 12.6 | 11 |
| 38 | `dc_w3_s0` | ds_dc0 | cuda | 3.50M | 4-32 | 5 | 5.1949 | **5.0706** | 12 | [8,16] | 11.3 | 11 |
| 39 | `dg_norm_s0` | ds_dc0 | cuda | 3.50M | 4-32 | 5 | 5.2629 | **5.1565** | 16 | [12,24] | 17.0 | 11 |
| 40 | `dc_control_s1` | ds_dc1 | cuda | 3.50M | 4-32 | 5 | 5.2547 | **5.1251** | 12 | [8,20] | 12.6 | 11 |
| 41 | `dc_w2_s1` | ds_dc1 | cuda | 3.50M | 4-32 | 5 | 5.2326 | **5.1136** | 12 | [8,20] | 12.6 | 11 |
| 42 | `dc_w3_s1` | ds_dc1 | cuda | 3.50M | 4-32 | 5 | 5.2148 | **5.0857** | 12 | [8,16] | 11.3 | 11 |
| 43 | `dg_norm_s1` | ds_dc1 | cuda | 3.50M | 4-32 | 5 | 5.2485 | **5.1274** | 20 | [12,32] | 19.6 | 11 |
| 44 | `dv_control_s0` | ds_div | cuda | 2.50M | 4-32 | 5 | 5.4693 | **5.3765** | 8 | [8,20] | 12.6 | 11 |
| 45 | `dv_lora_fixed0_s0` | ds_div | cuda | 2.50M | 4-32 | 5 | 5.3819 | **5.2734** | 12 | [8,20] | 12.6 | 11 |
| 46 | `dv_lora_r4_s0` | ds_div | cuda | 2.50M | 4-32 | 5 | 5.3648 | **5.2514** | 12 | [8,20] | 12.6 | 11 |
| 47 | `ds_od_control` | ds_operator_diversity_results | cuda | 2.50M | 4-32 | 5 | 5.4877 | **5.3874** | 12 | [8,20] | 12.6 | 11 |
| 48 | `ds_od_depth_gate` | ds_operator_diversity_results | cuda | 2.50M | 4-32 | 5 | 5.2047 | **5.0924** | 8 | [8,64] | 22.6 | 11 |
| 49 | `ds_od_lora_r4` | ds_operator_diversity_results | cuda | 2.50M | 4-32 | 5 | 5.3981 | **5.2863** | 12 | [8,20] | 12.6 | 11 |
| 50 | `pin_control_s0` | ds_pin2 | cuda | 2.50M | 4-32 | 5 | 5.4171 | **5.3052** | 12 | [8,20] | 12.6 | 11 |
| 51 | `pin_lora_b2_s0` | ds_pin2 | cuda | 2.50M | 4-32 | 5 | 5.3348 | **5.2237** | 12 | [8,16] | 11.3 | 11 |
| 52 | `xsa_control_s0` | ds_xsa | cuda | 2.50M | 4-32 | 5 | 5.3858 | **5.2851** | 8 | [8,16] | 11.3 | 11 |
| 53 | `xsa_on_s0` | ds_xsa | cuda | 2.50M | 4-32 | 5 | 5.2032 | **5.0689** | 12 | [8,16] | 11.3 | 11 |
| 54 | `expl_0.0` | explore_results | cuda | 6.00M | 4-32 | 5 | 4.9174 | **4.7704** | 8 | [8,16] | 11.3 | 11 |
| 55 | `expl_0.05` | explore_results | cuda | 6.00M | 4-32 | 5 | 4.9152 | **4.7646** | 8 | [8,16] | 11.3 | 11 |
| 56 | `expl_0.15` | explore_results | cuda | 6.00M | 4-32 | 5 | 5.0644 | **4.9530** | 8 | [8,16] | 11.3 | 11 |
| 57 | `expl_0.4` | explore_results | cuda | 6.00M | 4-32 | 5 | 5.5827 | **5.5604** | 8 | [2,32] | 8.0 | 11 |
| 58 | `gsweep_0.25` | gate_sweep_results | cuda | 6.00M | 4-32 | 5 | 4.8525 | **4.7133** | 8 | [8,16] | 11.3 | 11 |
| 59 | `gsweep_0.5` | gate_sweep_results | cuda | 6.00M | 4-32 | 5 | 4.9456 | **4.7906** | 8 | [8,16] | 11.3 | 11 |
| 60 | `gsweep_0.75` | gate_sweep_results | cuda | 6.00M | 4-32 | 5 | 4.8975 | **4.7480** | 8 | [8,16] | 11.3 | 11 |
| 61 | `gsweep_1.0` | gate_sweep_results | cuda | 6.00M | 4-32 | 5 | 4.9065 | **4.7749** | 8 | [8,16] | 11.3 | 11 |
| 62 | `gi_additive` | gated_inject_results | mps | 2.50M | 4-32 | 5 | 5.4952 | **5.4000** | 8 | [8,16] | 11.3 | 8 |
| 63 | `gi_gated` | gated_inject_results | mps | 2.50M | 4-32 | 5 | 5.4733 | **5.3730** | 8 | [8,16] | 11.3 | 8 |
| 64 | `gi_gated_a874` | gated_inject_results | mps | 2.50M | 4-32 | 5 | 5.7217 | **5.6470** | 8 | [8,16] | 11.3 | 8 |
| 65 | `kl_k1` | k_ladder_results | cuda | 2.50M | 4-32 | 1 | 5.8430 | **5.5783** | 16 | [12,24] | 17.0 | 11 |
| 66 | `kl_k2` | k_ladder_results | cuda | 2.50M | 4-32 | 2 | 5.6106 | **5.5081** | 12 | [8,20] | 12.6 | 11 |
| 67 | `kl_k3` | k_ladder_results | cuda | 2.50M | 4-32 | 3 | 5.4899 | **5.3877** | 12 | [8,20] | 12.6 | 11 |
| 68 | `kl_k5` | k_ladder_results | cuda | 2.50M | 4-32 | 5 | 5.4690 | **5.3576** | 8 | [8,16] | 11.3 | 11 |
| 69 | `kl_k8` | k_ladder_results | cuda | 2.50M | 4-32 | 8 | 5.3756 | **5.2819** | 8 | [8,16] | 11.3 | 11 |
| 70 | `kg_od_control_s1` | kg_rank8_results | cuda | 2.50M | 4-32 | 5 | 5.5236 | **5.4327** | 12 | [8,20] | 12.6 | 11 |
| 71 | `kg_od_lora_r8_s1` | kg_rank8_results | cuda | 2.50M | 4-32 | 5 | 5.4536 | **5.3593** | 12 | [8,20] | 12.6 | 11 |
| 72 | `kg_od_control_sw90` | kg_sw90_results | cuda | 2.50M | 4-32 | 5 | 5.6006 | **5.4821** | 16 | [8,24] | 13.9 | 11 |
| 73 | `kg_od_lora_r4_sw90` | kg_sw90_results | cuda | 2.50M | 4-32 | 5 | 5.4969 | **5.3649** | 16 | [8,24] | 13.9 | 11 |
| 74 | `od_control` | operator_diversity_results | mps | 2.50M | 4-32 | 5 | 5.4375 | **5.3391** | 12 | [8,16] | 11.3 | 8 |
| 75 | `od_depth_gate` | operator_diversity_results | mps | 1.56M | 4-32 | 5 | 5.7398 | **5.6518** | 12 | [8,16] | 11.3 | 8 |
| 76 | `od_lora_r2` | operator_diversity_results | mps | 2.50M | 4-32 | 5 | 5.5357 | **5.4332** | 12 | [8,16] | 11.3 | 8 |
| 77 | `od_lora_r4` | operator_diversity_results | mps | 2.50M | 4-32 | 5 | 5.4031 | **5.2877** | 12 | [8,16] | 11.3 | 8 |
| 78 | `rs_depth_init_s0` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.4481 | **5.3517** | 8 | [8,16] | 11.3 | 8 |
| 79 | `rs_depth_init_s1` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.5853 | **5.5049** | 8 | [4,16] | 8.0 | 8 |
| 80 | `rs_lambda1_s0` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.4840 | **5.3743** | 12 | [8,16] | 11.3 | 8 |
| 81 | `rs_lambda1_s1` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.5336 | **5.4387** | 8 | [8,16] | 11.3 | 8 |
| 82 | `rs_lambda2_s0` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.4950 | **5.3870** | 12 | [8,16] | 11.3 | 8 |
| 83 | `rs_lambda2_s1` | residual_scale_results | mps | 2.50M | 4-32 | 5 | 5.5737 | **5.4906** | 8 | [4,16] | 8.0 | 8 |
| 84 | `sand_P0R2C1` | sandwich_results | mps | 1.19M | 6-48 | 5 | 5.9537 | **5.9038** | 24 | [8,32] | 16.0 | 8 |
| 85 | `sand_P0R3C0` | sandwich_results | mps | 1.19M | 4-32 | 5 | 6.0213 | **5.9654** | 12 | [8,24] | 13.9 | 8 |
| 86 | `sand_P1R1C1` | sandwich_results | mps | 1.19M | 12-96 | 5 | 5.6203 | **5.5926** | 4 | [2,32] | 8.0 | 8 |
| 87 | `sand_P1R2C0` | sandwich_results | mps | 1.19M | 6-48 | 5 | 5.5886 | **5.5805** | 12 | [1,32] | 5.7 | 8 |
| 88 | `sc_clock` | scale_clock_results | mps | 0.31M | 4-32 | 5 | 6.8802 | **6.7845** | 8 | [4,16] | 8.0 | 8 |
| 89 | `sc_clock_sw90` | scale_clock_results | mps | 0.31M | 4-32 | 5 | 7.0758 | **7.0170** | 12 | [4,24] | 9.8 | 8 |
| 90 | `sc_ctrl` | scale_clock_results | mps | 2.50M | 4-32 | 5 | 5.5129 | **5.4202** | 8 | [8,16] | 11.3 | 8 |
| 91 | `sc_control_norm_s0` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.4692 | **5.3636** | 12 | [8,16] | 11.3 | 8 |
| 92 | `sc_control_norm_s1` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.6548 | **5.5789** | 8 | [4,16] | 8.0 | 8 |
| 93 | `sc_final_only_s0` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.4824 | **5.2654** | 8 | [4,8] | 5.7 | 8 |
| 94 | `sc_final_only_s1` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.2629 | **5.0624** | 8 | [8,8] | 8.0 | 8 |
| 95 | `sc_penalty_s0` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.2497 | **4.9975** | 8 | [8,12] | 9.8 | 8 |
| 96 | `sc_penalty_s1` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.3561 | **5.1165** | 8 | [8,12] | 9.8 | 8 |
| 97 | `sc_raw_s0` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.5594 | **5.3380** | 8 | [4,8] | 5.7 | 8 |
| 98 | `sc_raw_s1` | scale_control_results | mps | 2.50M | 4-32 | 5 | 5.7419 | **5.5363** | 4 | [4,8] | 5.7 | 8 |
| 99 | `center` | screening_results | mps | 0.89M | 4-32 | 5 | 6.7899 | **6.7723** | 4 | [4,32] | 11.3 | 8 |
| 100 | `fixed_loops16` | screening_results | mps | 1.19M | 16-16 | 5 | 6.7003 | **6.6440** | 4 | [4,8] | 5.7 | 8 |
| 101 | `inject_concat` | screening_results | mps | 0.89M | 4-32 | 5 | 6.7967 | **6.7565** | 2 | [2,32] | 8.0 | 8 |
| 102 | `inject_none` | screening_results | mps | 0.99M | 4-32 | 5 | 6.9513 | **6.9513** | 1 | [1,32] | 5.7 | 8 |
| 103 | `no_depth_init` | screening_results | mps | 0.99M | 4-32 | 5 | 6.9190 | **6.9139** | 8 | [1,32] | 5.7 | 8 |
| 104 | `no_state_renorm` | screening_results | mps | 0.99M | 4-32 | 5 | 6.0805 | **6.0281** | 8 | [4,16] | 8.0 | 8 |
| 105 | `truncate8` | screening_results | mps | 1.19M | 4-32 | 5 | 6.8291 | **6.7567** | 4 | [4,32] | 11.3 | 8 |
| 106 | `center_seed1` | second_seed_results | mps | 0.99M | 4-32 | 5 | 6.7519 | **6.7486** | 8 | [1,32] | 5.7 | 8 |
| 107 | `no_state_renorm_seed1` | second_seed_results | mps | 0.79M | 4-32 | 5 | 6.2782 | **6.2521** | 8 | [4,16] | 8.0 | 8 |
| 108 | `sd_dense_k5_s0` | supervision_depth_results | mps | 2.50M | 4-32 | 5 | 5.5611 | **5.4527** | 8 | [8,16] | 11.3 | 8 |
| 109 | `sd_dense_k5_s1` | supervision_depth_results | mps | 2.50M | 4-32 | 5 | 5.5369 | **5.4387** | 8 | [8,16] | 11.3 | 8 |
| 110 | `sd_terminal_k1_s0` | supervision_depth_results | mps | 2.50M | 4-32 | 1 | 5.7128 | **5.4699** | 16 | [12,24] | 17.0 | 8 |
| 111 | `sd_terminal_k1_s1` | supervision_depth_results | mps | 2.50M | 4-32 | 1 | 5.7552 | **5.4843** | 16 | [12,24] | 17.0 | 8 |
| 112 | `sup_concentrated24_32_s0` | supervision_results | mps | 2.50M | 24-32 | 5 | 5.6616 | **5.4584** | 16 | [12,24] | 17.0 | 8 |
| 113 | `sup_concentrated24_32_s1` | supervision_results | mps | 2.50M | 24-32 | 5 | 5.7089 | **5.5497** | 16 | [12,24] | 17.0 | 8 |
| 114 | `sup_shallow4_8_s0` | supervision_results | mps | 2.50M | 4-8 | 5 | 5.4034 | **5.3592** | 4 | [2,4] | 2.8 | 8 |
| 115 | `sup_shallow4_8_s1` | supervision_results | mps | 2.50M | 4-8 | 5 | 5.4249 | **5.3726** | 4 | [4,4] | 4.0 | 8 |
| 116 | `sup_uniform4_32_s0` | supervision_results | mps | 2.50M | 4-32 | 5 | 5.5699 | **5.4838** | 8 | [4,16] | 8.0 | 8 |
| 117 | `sup_uniform4_32_s1` | supervision_results | mps | 2.50M | 4-32 | 5 | 5.6001 | **5.5047** | 12 | [8,16] | 11.3 | 8 |
| 118 | `t1_mu40_term` | term_seed1_mu40_results | cuda | 2.50M | 32-48 | 1 | 6.5690 | **5.4901** | 40 | [40,48] | 43.8 | 13 |
| 119 | `trainL16` | train_at_L_results | cuda | 10.00M | 16-16 | 5 | 4.6177 | **4.4166** | 8 | [8,8] | 8.0 | 8 |
| 120 | `trainL2` | train_at_L_results | cuda | 10.00M | 2-2 | 5 | 4.4435 | **4.4229** | 2 | [2,2] | 2.0 | 8 |
| 121 | `trainL32` | train_at_L_results | cuda | 10.00M | 32-32 | 5 | 4.8377 | **4.4954** | 16 | [16,16] | 16.0 | 8 |
| 122 | `trainL4` | train_at_L_results | cuda | 10.00M | 4-4 | 5 | 4.4833 | **4.4297** | 4 | [2,4] | 2.8 | 8 |
| 123 | `trainL8` | train_at_L_results | cuda | 10.00M | 8-8 | 5 | 4.4875 | **4.3727** | 4 | [4,4] | 4.0 | 8 |
| 124 | `trainL16_s1` | train_at_L_seed1_results | cuda | 10.00M | 16-16 | 5 | 4.8421 | **4.6202** | 8 | [8,16] | 11.3 | 8 |
| 125 | `trainL2_s1` | train_at_L_seed1_results | cuda | 10.00M | 2-2 | 5 | 4.4832 | **4.4566** | 2 | [2,2] | 2.0 | 8 |
| 126 | `trainL32_s1` | train_at_L_seed1_results | cuda | 10.00M | 32-32 | 5 | 4.9496 | **4.6055** | 16 | [16,32] | 22.6 | 8 |
| 127 | `trainL4_s1` | train_at_L_seed1_results | cuda | 10.00M | 4-4 | 5 | 4.5449 | **4.4772** | 4 | [2,4] | 2.8 | 8 |
| 128 | `trainL8_s1` | train_at_L_seed1_results | cuda | 10.00M | 8-8 | 5 | 4.6951 | **4.5601** | 4 | [4,8] | 5.7 | 8 |
