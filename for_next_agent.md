# For the next agent — DDP study

Read in this order:
1. `.cursor/project.md` and `.cursor/instructions.md` — Arif's working style and global rules.
2. `task_log.md` — what's already in the repo, the PLSR replication status, the previous DL trials (set aside) and the reusable insights from them.
3. This file — your marching orders.

The PLSR replication is **done** (see `task_log.md`). The first DL push was **discarded** on 28 Apr 2026 because the framing (RBN vs baseline vs PLSR) was an apples-to-oranges architecture-vs-algorithm comparison. Code/results/markdowns from that push live in `archive/` for reference only — do not import from there.

## The hypothesis

**Tong et al. (2026) claim that the best preprocessing is region-dependent.** That conclusion is not algorithm-neutral. PLSR has no mechanism to push supervised signal back into the preprocessing stage, so the preprocessing must be picked manually from a fixed catalogue (snv/msc/sg/sgd) per-region. **When the downstream model is gradient-based (e.g. an MLP), gradients can flow back into a learnable preprocessing module, making the preprocessor itself data-driven.** We propose **DDP** (Data-Driven Preprocessing) — a learnable BN-only module — as one concrete instance of this idea, and aim to show that it is competitive with, or better than, the per-region winners under MLP, with the same single preprocessing choice for every region.

This is the only hypothesis. Forget H1A / H1B — they are dead.

## The experimental grid

Same 80/20 grouped+stratified split (seed=42, group=`Batch and labid`, strata=SOC quartile) is shared across every cell.

- **Preprocessing (7):** `none, snv, msc, sg, sgd, minmax, ddp`
- **Dataset (4):** `global, china, kenya, indonesia`
- **Algorithm (2):** `plsr` (Tong's; already populated for 6 preprocessings × 4 datasets in `results/per_cell/`), `mlp` (ours; to be populated for 7 × 4 = 28 cells)

PLSR cannot run on `ddp` (DDP is gradient-based; PLSR has no gradients). So the final reporting table has 24 PLSR cells + 28 MLP cells = 52 populated cells, with the 4 PLSR-on-DDP cells empty by design.

## The MLP

`Linear(n_features → 32) → ReLU → Dropout(p=0.3) → Linear(32 → 1)`. n_features = 1763.

- Dropout is included **only** to regularise the low-data regime (esp. Indonesia n=188). It is not part of the BN-efficacy claim. Same `p=0.3` everywhere, every preprocessing row, both DDP stages.
- Training: full-batch (batch_size = full train size) for **all** preprocessing rows. This keeps the protocol uniform across cells and matches the BN-stability finding from the previous phase (mini-batch BN is noisy at n < ~200).
- Optimiser: **Adam**, lr=1e-3, weight_decay=0, MSE loss.
- Epochs: **fixed 500** (no early stopping, no validation set carve-out). Same budget for every cell.
- Seed: 42.
- Loss: MSE on raw SOC.

The MLP architecture and hyperparameters are identical for every preprocessing row, so any difference in test metrics is attributable to the preprocessing.

## DDP — the data-driven preprocessing module

DDP is a single PyTorch module: `nn.BatchNorm1d(n_features)` (no head, no nonlinearity, no extra layers). It has 2·n_features learnable parameters (γ, β) and 2·n_features non-learnable running statistics (running mean, running variance).

DDP is fit by **two-stage training** so it slots into the same column as snv/msc/etc. (i.e. "preprocessing fitted on train, applied to train+test, then a model is trained on the transformed data"):

1. **Stage 1 — fit the preprocessor.** Train `DDP → MLP` end-to-end on raw train spectra and SOC, full-batch, 500 epochs, Adam lr=1e-3, dropout p=0.3, seed=42, MSE loss. This is the only place DDP's γ, β, running mean, running variance get updated. **Record stage-1 SOC test metrics** (RMSE, R², MBD, RPIQ on the held-out 20% test set, evaluated with DDP+MLP in eval mode). These metrics are part of the experimental output — Arif's prior is that stage-1 and stage-2 SOC metrics will come out very close, and that's worth confirming with numbers.
2. **Stage 2 — freeze and re-apply.** Put DDP into `eval()` mode (so it uses the running mean/variance, *not* the current batch's statistics). Discard the stage-1 MLP head. Run train and test spectra through frozen DDP to get `ddp_train_X` and `ddp_test_X`. Then train a **fresh** MLP (same architecture, same hyperparameters as every other preprocessing row, freshly seeded with seed=42) on `ddp_train_X` and SOC, full-batch, 500 epochs. Evaluate on `ddp_test_X` and SOC. Record **stage-2 SOC test metrics**. These are the canonical DDP metrics that go into the comparison table alongside snv/msc/etc.

Why this two-stage protocol matters for the framing:
- It forces DDP to look procedurally identical to every other preprocessing row (fit on train, apply, then a fresh model is trained on the transformed data).
- The frozen DDP is a fixed per-band affine, so MLP-on-DDP-data has the same representational capacity as MLP-on-raw-data. **Any improvement therefore comes from preprocessing-style benefits (conditioning, scale alignment), not from extra model capacity.** That is exactly the claim we want to make.
- Recording stage-1 metrics in addition to stage-2 metrics lets us check (and report) that the two-stage trick did not cost us SOC performance vs. joint training. If stage-1 ≈ stage-2 across all 4 datasets, the framing is clean.

## Locked decisions (Arif, 28 Apr 2026)

1. PLSR stays in the comparison table as a parallel reference column. PLSR cells are already populated for 6 preprocessings × 4 datasets in `results/per_cell/`; do **not** rerun the PLSR pipeline.
2. Stage-1 and stage-2 MLPs are architecturally identical and use identical hyperparameters. Both stage-1 and stage-2 SOC test metrics are recorded.
3. Dropout fixed at p=0.3 across all 7 preprocessing rows (and both DDP stages). No per-cell tuning.
4. **No early stopping. Universal 500 epochs for every cell.** This is non-negotiable for now — the prior phase's epoch-tuning rabbit hole is not to be re-entered.
5. Old DL artefacts (rbn, rbnd, rbnr, mmbnd, baseline, cnn, transformer variants and all their results) live in `archive/`. Do not consume from there. Do not resurrect any of those models.
6. Old markdown files (`experiment_spec.md`, `our_task_log.md`, the previous `for_next_agent.md`) are in `archive/md/`. The current canonical references are `task_log.md` and this file.
7. The DDP hypothesis is the only hypothesis. Do not introduce sub-hypotheses (no H1A / H1B / etc.) unless Arif asks for them.

## Concrete deliverables you should create

These are the files to produce. Names are suggestions — feel free to rename if there's a better convention, but stay consistent with the existing PLSR naming (`train_*.py`, `model_*.py`).

1. **`model_mlp.py`** — defines `MlpSocAnn(n_features)` with the architecture above (`Linear → ReLU → Dropout(0.3) → Linear → squeeze`). Include `count_learnable_parameters()` for parity with the existing PLSR code.
2. **`model_ddp.py`** — defines `DdpPreprocessor(n_features)`, a tiny module with a single `nn.BatchNorm1d(n_features)`. Forward returns the BN output. Include a method or convention for "freeze in eval mode and apply to a numpy/tensor batch" so stage 2 can call it as a pure preprocessor.
3. **`train_ddp_experiment.py`** — the runner. For each (dataset × preprocessing) cell:
   - For preprocessing ∈ {none, snv, msc, sg, sgd, minmax}: load the matching `data/preprocessed/{dataset}_{preprocessing}_{train,test}.csv`, train MLP fresh on it for 500 epochs full-batch, evaluate, record metrics.
   - For preprocessing == `ddp`: load `data/preprocessed/{dataset}_none_{train,test}.csv` (raw). Stage 1: train `DDP → MLP` end-to-end, record stage-1 test metrics. Stage 2: freeze DDP, transform both splits, train fresh MLP on transformed train, record stage-2 test metrics. Save **both** stage-1 and stage-2 metrics.
   - All cells: use seed=42, full-batch, lr=1e-3, dropout=0.3, 500 epochs, Adam, MSE.
   - Output one JSON per cell in `results/ddp_experiment/cells/{dataset}_{preprocessing}_mlp.json` with the same metric schema as `results/per_cell/` (rmse, r2, mbd, rpiq, n) plus a `configuration` block. For the `ddp` cells the JSON should carry both `stage1_test_metrics` and `stage2_test_metrics`. After all cells run, write a summary CSV `results/ddp_experiment/cell_results.csv` (long format).
4. **A reporting/comparison script** (e.g. `report_ddp_experiment.py`) that joins the MLP-side results with the PLSR-side results from `results/per_cell/` and prints (a) a wide table preprocessing × dataset for `mlp_test_rmse`, (b) the same for `plsr_test_rmse`, and (c) a per-region "winner" highlight. Should also output a side-by-side `results/comparison_table.csv` for the paper.

## Operational gotchas (carried over)

1. **PowerShell on Windows** is the default shell. Heredocs (`cat <<EOF`) do not work — use temp files or `git commit -F file.txt` for multiline strings.
2. **`$env:VENV_PY`** holds the path to the venv's `python.exe`. Use `& $env:VENV_PY script.py` to invoke Python from a Shell call.
3. **Auto-commit hook** is active. Some tool periodically auto-commits the working tree. `results/` is gitignored; new heavy result folders should be too. Check `.gitignore` before writing new artefact directories.
4. **Stdout is buffered** for long Python scripts run via Shell. Watch progress by tailing the per-cell JSON output folder, not the terminal text.
5. **Don't import from `archive/`.** It's a graveyard, not a library. If you need a helper that lived in `archive/code/train_pbn_experiment.py` (e.g. `compute_metrics_dictionary`, `extract_spectra_and_target`), copy it into a new clean utilities module rather than importing across the archive boundary.
6. The shared utilities you'll likely want to recreate (one-time): `extract_spectra_and_target(df)`, `compute_rmse / r2 / mbd / rpiq / metrics_dictionary`, `reset_seeds(seed)`. They were ~50 lines total in `archive/code/train_pbn_experiment.py` and are simple enough to rewrite cleanly.

## What to do first (concrete starter checklist)

1. Read `task_log.md` end-to-end. Pay particular attention to the "Previous DL trials and findings" section — those traps are documented so you don't fall into them again.
2. Skim `archive/code/model_rbnd_ann.py` to see the exact architecture we settled on (it is structurally `DDP → MLP` with dropout). Do **not** import or reuse the file; rebuild cleanly under the new naming (`model_mlp.py` + `model_ddp.py`).
3. Skim one PLSR cell JSON (`results/per_cell/global_none.json`) so the new MLP cell JSONs match the schema (same metric keys, same nesting).
4. Write `model_mlp.py` and `model_ddp.py`. Confirm parameter counts: MLP at n_features=1763 has 1763·32 + 32 + 32·1 + 1 = **56,481** learnable params; DDP+MLP adds 2·1763 = **3,526** more for γ, β = **60,007** total in stage 1.
5. Write `train_ddp_experiment.py`. Implement the 28-cell loop (4 × 7), with the special-case branching for `ddp` doing the two-stage training. Print per-cell progress lines like the existing PLSR runner does. Each cell should be skip-if-exists idempotent (look at `results/ddp_experiment/cells/`).
6. Run a single cell first as a smoke test (e.g. `indonesia / none`) to confirm the loop works and the JSON shape is right. Then run all 28 cells. Indonesia full-batch at 500 epochs should be sub-second per cell on CPU; total runtime is on the order of minutes.
7. Write the reporting script. Generate the comparison table.
8. **Then stop and report to Arif** — show him the comparison table and per-region rankings before doing anything fancier (multi-seed runs, statistical tests, plots beyond the table). Let Arif decide the next move.

## What to *not* do (without asking Arif)

1. Do not introduce early stopping, validation splits, or any kind of LR/epoch tuning. The 500-epoch / lr=1e-3 / dropout=0.3 contract is locked.
2. Do not add new model architectures (CNN / transformer / deeper MLP / attention). The MLP is locked.
3. Do not change the train/test split, group key, stratification, or seed. They are project-wide constants.
4. Do not run multi-seed bands until Arif has seen the single-seed comparison table.
5. Do not try to "fix" the PLSR results to get closer to paper Table 1. That avenue is closed (see `task_log.md`).
6. Do not mark `ddp` as the winner in the comparison table; just report the numbers. Arif will read them and decide the framing for the writeup.

## What Arif wants from you (style reminders)

1. "Less text, my eyes hurt." Default to terse. Numbered lists, no decorative tables, no preamble.
2. Update `task_log.md` after every meaningful step. The reproduction recipe at the top must stay copy-pasteable.
3. Decisions taken without asking get logged with a one-liner reason in `task_log.md` "Decisions on ambiguities".
4. When in Ask mode and Arif requests code, do not write code — remind him to switch to Agent mode. (See `.cursor/instructions.md`.)
5. When Arif's request is ambiguous, pick the best decision and log it; do not over-use `AskQuestion`.
