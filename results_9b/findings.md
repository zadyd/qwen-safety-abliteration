# Qwen3.5-9B Abliteration Findings

## Baseline
- Baseline refusal rate: 100.0%
- Baseline ASR: 0.0%

## Key Findings

### Batch Abliteration Results (multi-layer groups)

| Group        | Layers               | ASR    | Refusal Drop |
|--------------|----------------------|--------|--------------|
| full_model    | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 100.0% |     +100.0% |
| top_half      | [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 50.0% |      +50.0% |
| mid_top8      | [16, 17, 18, 19, 20, 21, 22, 23] | 40.0% |      +40.0% |
| last_12       | [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 20.0% |      +20.0% |
| last_8        | [24, 25, 26, 27, 28, 29, 30, 31] |  0.0% |       +0.0% |
| last_6        | [26, 27, 28, 29, 30, 31] |  0.0% |       +0.0% |
| last_4        | [28, 29, 30, 31]     |  0.0% |       +0.0% |

**Most effective group**: `full_model` — ASR=100.0%, refusal_drop=+100.0%
Active layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

### Critical Refusal Layers (refusal drop > 5%)

- **Layer 0** (LINEAR_ATTN): refusal drop = +100.0%
- **Layer 1** (LINEAR_ATTN): refusal drop = +100.0%
- **Layer 2** (LINEAR_ATTN): refusal drop = +100.0%
- **Layer 11** (FULL_ATTN): refusal drop = +100.0%
- **Layer 14** (LINEAR_ATTN): refusal drop = +100.0%
- **Layer 10** (LINEAR_ATTN): refusal drop = +90.0%
- **Layer 3** (FULL_ATTN): refusal drop = +80.0%
- **Layer 6** (LINEAR_ATTN): refusal drop = +80.0%
- **Layer 9** (LINEAR_ATTN): refusal drop = +80.0%
- **Layer 12** (LINEAR_ATTN): refusal drop = +80.0%
- **Layer 4** (LINEAR_ATTN): refusal drop = +70.0%
- **Layer 7** (FULL_ATTN): refusal drop = +50.0%
- **Layer 8** (LINEAR_ATTN): refusal drop = +50.0%
- **Layer 13** (LINEAR_ATTN): refusal drop = +50.0%
- **Layer 5** (LINEAR_ATTN): refusal drop = +40.0%
- **Layer 17** (LINEAR_ATTN): refusal drop = +40.0%
- **Layer 15** (FULL_ATTN): refusal drop = +20.0%
- **Layer 16** (LINEAR_ATTN): refusal drop = +20.0%
- **Layer 18** (LINEAR_ATTN): refusal drop = +20.0%
- **Layer 19** (FULL_ATTN): refusal drop = +10.0%

### Full Attention Layers (n=8)
Indices: [3, 7, 11, 15, 19, 23, 27, 31]

### Linear Attention Layers (n=24)

## Per-Layer Detailed Results

| Layer | Type       | ASR     | Refusal Drop |
|-------|------------|---------|--------------|
|     0 | LINEAR     |  100.0% |      +100.0% |
|     1 | LINEAR     |  100.0% |      +100.0% |
|     2 | LINEAR     |  100.0% |      +100.0% |
|     3 | FULL       |   80.0% |       +80.0% |
|     4 | LINEAR     |   70.0% |       +70.0% |
|     5 | LINEAR     |   40.0% |       +40.0% |
|     6 | LINEAR     |   80.0% |       +80.0% |
|     7 | FULL       |   50.0% |       +50.0% |
|     8 | LINEAR     |   50.0% |       +50.0% |
|     9 | LINEAR     |   80.0% |       +80.0% |
|    10 | LINEAR     |   90.0% |       +90.0% |
|    11 | FULL       |  100.0% |      +100.0% |
|    12 | LINEAR     |   80.0% |       +80.0% |
|    13 | LINEAR     |   50.0% |       +50.0% |
|    14 | LINEAR     |  100.0% |      +100.0% |
|    15 | FULL       |   20.0% |       +20.0% |
|    16 | LINEAR     |   20.0% |       +20.0% |
|    17 | LINEAR     |   40.0% |       +40.0% |
|    18 | LINEAR     |   20.0% |       +20.0% |
|    19 | FULL       |   10.0% |       +10.0% |
|    20 | LINEAR     |    0.0% |        +0.0% |
|    21 | LINEAR     |    0.0% |        +0.0% |
|    22 | LINEAR     |    0.0% |        +0.0% |
|    23 | FULL       |    0.0% |        +0.0% |
|    24 | LINEAR     |    0.0% |        +0.0% |
|    25 | LINEAR     |    0.0% |        +0.0% |
|    26 | LINEAR     |    0.0% |        +0.0% |
|    27 | FULL       |    0.0% |        +0.0% |
|    28 | LINEAR     |    0.0% |        +0.0% |
|    29 | LINEAR     |    0.0% |        +0.0% |
|    30 | LINEAR     |    0.0% |        +0.0% |
|    31 | FULL       |    0.0% |        +0.0% |

## Top Layers by Refusal Direction Score

| Rank | Layer | Score |
|------|-------|-------|
|    1 | 31    | 37.0000 |
|    2 | 30    | 29.6052 |
|    3 | 29    | 26.3109 |
|    4 | 28    | 24.2309 |
|    5 | 27    | 21.5859 |
|    6 | 26    | 20.7763 |
|    7 | 25    | 18.9379 |
|    8 | 24    | 17.5730 |
|    9 | 23    | 15.8891 |
|   10 | 22    | 14.6277 |

## Conclusions

- TBD: analyze which attention types dominate the refusal signal
- TBD: identify whether refusal is concentrated in later layers or mid-layers
- TBD: compare with Qwen3.5-2B findings (layers 6-14) and Qwen3.5-27B (layers 48-63)