# Biohub — Cell Tracking During Development

Write-up for the Kaggle competition [Biohub: Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). The task is to detect cells in 3D+time fluorescence microscopy, link them across frames, and reconstruct cell divisions (lineage tracking).

| | Public | Private |
|---|---:|---:|
| Final submissions (XH00, XH04) | 0.953 (rank 673 / 4,020) | **0.917** |
| Best private among our submissions (XH08) | 0.953 | 0.920 |

## Notebook: [`biohub_cell_tracking_report.ipynb`](biohub_cell_tracking_report.ipynb)

The notebook is written in Korean and ships with all outputs and figures.

1. **Data analysis** (199 training movies)
   - Movies are `T × Z × Y × X` = 100 × 64 × 256 × 256. Z voxels (1.625 µm) are 4× longer than XY voxels (0.40625 µm), so distances must be computed in µm.
   - Intensity dynamic range varies about 10× across movies.
   - Ground truth is very sparse: annotated nodes are about 3% (median) of the cells in a movie.
   - Cells move a median of 1.8 µm per frame; only 3.6% of steps exceed 6 µm.
   - Divisions are rare (0–5 per movie).
2. **Metric**: adjusted edge Jaccard + 0.1 × division Jaccard
3. **Submission history**: Public vs Private for all 37 scored submissions, and our position on the leaderboard
4. **Pipeline**: Temporal 3D U-Net + Node Transformer → ILP → motion relinking, gap repair, division repair, smoothing
5. **Post-processing experiments XH00–XH08**: one change at a time on a byte-exact reproduction of a strong public pipeline
6. **Error diagnosis on all 199 training movies**: official score after every post-processing stage
7. **Event-level tracing**: why correct links and divisions were lost
8. **Our own baseline training**: learning curves
9. **Lessons and next steps**

## Key findings

- **The first motion relinking pass did the most damage.** It repaired 493 correct links but broke 704. Every broken link we traced was the association model's top-ranked choice: a geometric rule overwrote what the learned model got right.
- **Divisions were the weakest part.** Division Jaccard was 0.11 (precision 27%, recall 16%), and all of it came from a post-hoc repair step. The most common miss: the second daughter had already been linked to another parent.
- **53% of missed links had both cells detected.** Link selection, not the detector, was the main bottleneck.
- **Public ties hid Private differences.** Public and Private rankings were strongly correlated overall (Spearman ρ = 0.95), but submissions tied at Public 0.953 ranged from 0.917 to 0.920 on Private.

## What I learned

- **Reproduce before improving.** A byte-exact reproduction of the strongest public pipeline gave a trustworthy control. Every later comparison was measured against it.
- **Score every stage, not just the final output.** Re-scoring all 199 training movies after each post-processing step showed exactly where points were won and lost. A single final number would never have shown that one step broke more links than it fixed.
- **Fix the bottleneck instead of tuning thresholds.** Eight post-processing variants all landed on the same Public score. The real problems (motion relinking overriding the model, links and divisions competing in the wrong order) needed a change to how links are selected, not another threshold.
- **Three-decimal Public ties are not a basis for choosing final submissions.** Large gains carried over to Private; tiny ones did not. Final picks need evidence beyond the Public leaderboard.
- **With sparse labels, unmatched predictions are not false positives.** Most cells are never annotated, so evaluation and training losses must treat "unlabeled" differently from "wrong".
- **Plan compute for the last week.** Our weekly GPU quota ran out just before the deadline, so no new GPU candidate could run. A CPU fallback also timed out on the larger hidden test set. In a code competition, runtime on the hidden data is part of the solution.

## Reflections

Reaching 0.953 by carefully reproducing a public pipeline came quickly. Pushing past it was the hard part. We spent a lot of time on audits, safety checks and diagnosis. That work paid off: it pinpointed the exact failure modes. But it left too little time to build the model changes that would fix them. The diagnosis was right, and the timing was wrong.

The strict submission checks prevented several broken submissions, but they also slowed iteration near the end. Next time I would freeze the evaluation setup early, reserve GPU time for the final week, and start on the biggest diagnosed bottleneck as soon as it is found, instead of first exploring more post-processing variants.

## Running

The notebook is saved with all outputs. Re-running it requires the competition data and intermediate experiment artifacts (submission CSVs, ground-truth graph indexes, per-stage diagnosis results), which are not included here, per the competition rules and for size.

```bash
pip install numpy pandas matplotlib jupyter
```

## Acknowledgements

The final pipeline builds on public Kaggle work:

- Anvith Pothula — X138 / Harmonic Fusion notebooks and [`biohub-v1284-head-s075`](https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075)
- pilkwang — [`biohub-temporal-unet3d-seed314159-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1), [`biohub-deepcenter-unet3d-center-prior-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1), [`biohub-tracking-support-pack-50ep-v1`](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1)
- Official metric: [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)
