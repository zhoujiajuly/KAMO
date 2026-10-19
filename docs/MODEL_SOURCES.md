# Model sources and adaptations

The mean-prediction backbones in `kamo_experiments/vendor/` are
the authors' official Python files, copied without architectural edits from
the exact commits below. Their full Apache-2.0 licenses are included alongside
the files. The wrapper is implemented separately in `kamo_experiments/models.py`.
There are no extra neural-network dependencies beyond PyTorch and NumPy.

| Method | Official publication | Repository and pinned commit | Local source |
|---|---|---|---|
| DLinear / NLinear | Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023 | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6), `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `vendor/dlinear.py`, `vendor/nlinear.py` |
| CycleNet | Lin et al., *CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns*, [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/bfe7998398779dde03cad7a73b1f81b6-Paper-Conference.pdf) | [ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet/tree/d807e51fc2dcd143885ee639d97965a7ab0926f4), `d807e51fc2dcd143885ee639d97965a7ab0926f4` | `vendor/cyclenet.py` |
| TQNet | Lin et al., *Temporal Query Network for Efficient Multivariate Time Series Forecasting*, [ICML 2025, PMLR 267](https://proceedings.mlr.press/v267/lin25e.html) | [ACAT-SCUT/TQNet](https://github.com/ACAT-SCUT/TQNet/tree/15e19cb23483ed52398566c4baa959168cfffa57), `15e19cb23483ed52398566c4baa959168cfffa57` | `vendor/tqnet.py` |

TQNet supplies a genuine 2025 published comparison. It models cross-variable
relations using periodic learnable queries and an attention/MLP predictor.
CycleNet supplies a recent lightweight periodic baseline. These citations do
not establish that a method will perform well on these corruption conditions.

## Exact departures from the original point-forecast protocols

1. **Gaussian adaptation.** Mean output is the original official backbone.
   Every added model receives the same channel-shared `Linear(seq_len,pred_len)`
   standard-deviation head, followed by softplus and a positive floor. Its input
   is observed-history centered/scaled data; its output is rescaled to the data
   units using observed-history variance. No future values enter this head.
   The wrapper returns a diagonal Gaussian. The training runner chooses the
   common probabilistic loss and selection rule. These experiments must be
   called **probabilistic adaptations using official mean backbones**, not
   reproductions of the paper's original MSE-training results.
2. **Missing inputs.** The wrapper intersects the supplied mask with finite-value
   locations. Missing entries use the observed mean of the same input window;
   an entirely missing variable uses zero. Observed values are preserved.
   This wrapper is shared across all four added methods. The original mean
   prediction is exactly preserved on complete finite inputs; automated tests
   verify this equality against the vendored backbones.
3. **DLinear is corrected.** This implementation really uses the official
   moving-average trend/seasonal decomposition, with two separate linear maps.
   NLinear uses last-value centering and one linear map. Earlier weights named
   `dlinear` are not compatible and must not be relabeled or silently loaded.
4. **Cycle phase.** `cycle_index` is the global integer start index of the input
   history. It must travel with each sample through shuffling, validation, test,
   and corruption evaluation. Test/validation must not reset the phase at the
   split boundary. A globally fixed offset is harmless only if used consistently
   across every split. The wrapper raises an error if this metadata is absent.
5. **Official details retained.** CycleNet defaults to its two-layer ReLU MLP
   residual backbone. TQNet retains its four-head attention and upstream fixed
   attention dropout of 0.5; searched `dropout` controls the output projector.
   Both use upstream per-window reversible normalization. TQNet history length
   must be divisible by four. DLinear/NLinear use their original global-dataset
   standardized inputs without extra RevIN on their mean path.

## Search and period configuration

`new_model_search_space(name,dataset)` supplies effective model-specific
hyperparameters; the runner allocates the same number of trials under the
same validation criterion and maximum training budget. Equal search budgets
do not mean that all architectures have the same parameter count.

- DLinear/NLinear: the factory supports shared or variable-specific linear weights.
  The shipped formal runner deliberately fixes shared weights (`linear_individual=False`)
  to retain a genuinely lightweight baseline and searches all four learning rates
  times three weight decays (12 effective candidates). No dropout/hidden-width
  placebo trials are counted.
- CycleNet: learning rate, weight decay, MLP hidden width and declared cycle.
  It has no dropout layer, so dropout must not consume a search trial.
- TQNet: learning rate, weight decay, hidden width, output dropout and cycle.
- Conventional declared cycles: ETTh1/ETTh2 24 (hourly day), Weather 144
  (10-minute day), Electricity 168 (hourly week).
- Exchange: candidates 5 and 20 describe financial business-day heuristics,
  not a proven periodic structure. Select using validation only within the
  same allotted trial budget and report the chosen value. Other datasets require
  an explicitly justified period; do not infer one from the test targets.

The four new models add trainable variance parameters that must be counted in
parameter, MAC/FLOP, timing, and memory measurements. Their means are validated
against official code, but numerical results still need the user's server runs.

## Quick verification

From the package root:

```bash
python -m unittest discover -s tests -p test_models.py -v
```

Tests cover shape, finite forward/backward values, positive scales, exact
upstream mean equality, missing-value invariance, cycle metadata requirements,
batch-order invariance, DLinear decomposition and individual linear heads.
Verified locally with PyTorch 1.11.0 CPU; no long model training was performed.

## Locally implemented comparisons

This is a code/provenance audit, not a fresh benchmark reproduction. The inherited
implementations in `kamo/baselines.py` are retained for continuity. They are
local probabilistic implementations and must not be presented as unchanged
official-code replications merely because a paper name is used as a run key.

| Run key | Structure observed in the inherited source | Accurate reporting label / limitation |
|---|---|---|
| `patchtst` | Channel-independent patch embedding, a PyTorch pre-norm TransformerEncoder with GELU, LayerNorm, learned patch positions and flattened Gaussian head | Local probabilistic PatchTST adaptation; mean/backbone exact equality to the official implementation has not been established |
| `itransformer` | Linear embedding of each variable's history, added trainable variable-ID embedding, PyTorch pre-norm TransformerEncoder and Gaussian head | Local probabilistic iTransformer adaptation; the variable-ID embedding and exact normalization/attention path need disclosure |
| `timemixer` | Pooled resolutions, separate moving-average seasonal/trend MLPs within each scale, independent horizon heads and learned mixture weights | Compact TimeMixer-style multiscale control; it is not the full official past-decomposable-mixing/future-multipredictor implementation |
| `vpnet` | Patch encoder/decoder, depthwise 2-D variate/patch convolution, latent forecast MLP, separate variance projection and reconstruction loss | Local probabilistic VPNet adaptation; source comments claim official-design ancestry, but this package does not certify upstream checkpoint/code equivalence |
| `timemixer_robust` | The inherited compact TimeMixer-style control preceded by KAMO's parameter-free causal median/MAD robustifier | **New direct architectural control**, not an independently published robust forecasting method and not an official robust TimeMixer variant |
| `patchtst_robust`, `itransformer_robust` | Same parameter-free robustifier preceding the corresponding local probabilistic adaptation | Authors' fixed-preprocessing controls, distinct from published point-forecast baselines |

The TimeMixer-style control tests whether fixed robust preprocessing plus a
multiscale predictor can explain the observed gains. Its usefulness for this
mechanism question does not make it a replacement for an exact recent published
method comparison. The added official DLinear/CycleNet/TQNet mean backbones help
address that separate need; preserve both kinds of evidence and label them.

Widths 32, 64 and 128 are compatible with these code paths and four attention
heads. For TQNet, it is **history length**, not hidden width, that must be
divisible by four. VPNet additionally requires history and horizon lengths
divisible by its patch length. Search choices must satisfy these constraints.

For the explicitly lightweight linear comparison, prefer shared
`linear_individual=False`: 4 learning rates × 3 weight-decay values already
give 12 effective optimizer candidates. Variable-specific DLinear can become
large: at 321 variables, history 96 and horizon 720, its two mean maps alone
contain about 44 million weights, versus about 138 thousand shared weights.
An individual configuration is legitimate if declared, but should not then be
described as universally lightweight. TQNet/iTransformer attention memory also
grows quadratically in variable count; reduce the microbatch uniformly for
high-dimensional runs and record the effective batch/accumulation protocol.
