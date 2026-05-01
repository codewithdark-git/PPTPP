# PPT++: Efficient Token Pruning and Pooling for Vision Transformers

PPT++ is a token-efficient extension of PPT for DeiT- and LV-ViT-style backbones. The project studies how to reduce the computational cost of vision transformers while preserving the information carried by image tokens, using a combination of pruning, squeezing, threshold adaptation, and spatial-aware pooling.

The main objective is not to redesign the backbone itself, but to introduce a lightweight compression policy that can be injected into an existing model with minimal changes.

## Brief Theoretical Overview

Vision transformers operate on a sequence of patch tokens. At intermediate layers, many tokens become redundant for classification, but naive removal can discard useful signal. PPT++ treats token reduction as a structured compression problem with two complementary operations:

1. Pruning removes low-utility tokens based on attention-aware token scores.
2. Pooling merges tokens when the layer-level uncertainty suggests that consolidation is preferable to deletion.

At each compression layer, the module computes token importance scores, measures score variance, and uses a threshold rule to choose between pruning and pooling. In the current implementation, the decision rule is applied per sample and per compression layer.

## Contributions

| Contribution | Core idea | Where it is applied |
|---|---|---|
| Token Squeezing | Preserve information from removed tokens by fusing them back into retained tokens with cosine-similarity weights. | `ppt_pp/compression/pruner.py` |
| Adaptive Per-Layer Thresholding | Replace a single global threshold with layer-specific EMA statistics, using $\tau_l = \mu_l + \alpha \sigma_l$. | `ppt_pp/compression/threshold.py` |
| Spatial-Aware Bipartite Soft Matching | Extend BSM with a mixture of visual and spatial similarity for merging tokens. | `ppt_pp/compression/pooler.py` |

## How PPT++ Is Applied To The Baseline

PPT++ is injected into an existing ViT backbone through hook-based integration. The backbone remains the same; the compression policy is attached at selected layers.

The operational flow is:

1. Extract attention, key, and value tensors from the baseline transformer block.
2. Score image tokens using the class-token attention signal and value magnitude.
3. Compute score variance for the current sample.
4. Compare the variance to the chosen threshold for that layer.
5. If the sample is assigned to pruning, keep the top-$K$ tokens and optionally squeeze pruned-token information back into the retained set.
6. If the sample is assigned to pooling, merge tokens using standard BSM or spatial-aware BSM.
7. Reassemble the compressed token sequence and continue the forward pass.

In short, the baseline model is preserved, while PPT++ controls token reduction at the selected compression layers.

## Baseline Integration

| Baseline family | Compression layers | Default tokens removed per layer | Threshold |
|---|---:|---:|---:|
| DeiT-Ti / DeiT-S / DeiT-B | `[3, 6, 9]` | `[16, 16, 16]` | `7e-5` |
| LV-ViT-S | `[4, 8, 12]` | `[50, 50, 50]` | `5e-4` |
| LV-ViT-M | `[5, 10, 15]` | `[50, 50, 50]` | `5e-4` |

These defaults correspond to the presets defined in `ppt_pp/compression/ppt_module.py` and can be adapted for other backbones.

## Representative Results

| Model | Baseline Top-1 (%) | PPT++ Top-1 (%) | Baseline FLOPs (G) | PPT++ FLOPs (G) | FLOPs ↓ |
|---|---:|---:|---:|---:|---:|
| DeiT-S | 79.8 | 79.8 | 4.6 | 2.9 | 37.0% |
| DeiT-S, fine-tuned | 79.8 | 80.1 | 4.6 | 2.9 | 37.0% |
| LV-ViT-S | 83.3 | 83.0 | 6.6 | 4.6 | 30.3% |
| LV-ViT-S, fine-tuned | 83.3 | 83.3 | 6.6 | 4.6 | 30.3% |

| Compression schedule | Tokens removed per layer | FLOPs ↓ | Notes |
|---|---|---:|---|
| Uniform | `[16, 16, 16]` | 10.8% | Balanced compression across stages |
| Pyramid | `[24, 16, 8]` | 13.0% | Stronger early reduction |
| Front-heavy | `[32, 12, 4]` | 14.6% | Most aggressive early compression |

## Repository Structure

- `main.py` - CLI entrypoint and experiment runner.
- `setup.md` - local setup instructions and experiment commands.
- `ppt_pp/compression/scorer.py` - token scoring.
- `ppt_pp/compression/threshold.py` - fixed and adaptive thresholds.
- `ppt_pp/compression/pruner.py` - top-k pruning and token squeezing.
- `ppt_pp/compression/pooler.py` - BSM and spatial-aware pooling.
- `ppt_pp/compression/ppt_module.py` - hook-based compression wrapper.
- `ppt_pp/models/` - model builders and LV-ViT implementations.
- `ppt_pp/experiments/run_all.py` - experiment routines used for the reported tables.
- `ppt_pp/utils/metrics.py` - FLOPs, throughput, and quality metrics.

## Reproducibility

For environment setup and local execution, see [setup.md](setup.md). The experiment scripts in `ppt_pp/experiments/run_all.py` and the LaTeX source in `ppt_pp_paper_refined.tex` provide the closest path to reproducing the tables in this repository.

## Citation

If you use PPT++ in your work, please cite the project or paper once the final bibliographic entry is available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## References

- **PPT**: [arXiv:2310.01812](https://arxiv.org/abs/2310.01812) - Wu et al., 2023
- **ToMe**: [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) - Bolya et al., ICLR 2023
- **TPS**: [arXiv:2304.10716](https://arxiv.org/abs/2304.10716) - Wei et al., CVPR 2023
- **LV-ViT**: [arXiv:2104.10858](https://arxiv.org/abs/2104.10858) - Jiang et al., NeurIPS 2021
- **DeiT**: [arXiv:2012.12877](https://arxiv.org/abs/2012.12877) - Touvron et al., ICML 2021
