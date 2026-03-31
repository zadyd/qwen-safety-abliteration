# Qwen3.5-2B Abliteration Findings

## Baseline
- Baseline refusal rate: 100.0%
- Baseline ASR: 0.0%

## Key Findings

### Batch Abliteration Results (multi-layer groups)

| Group        | Layers               | ASR    | Refusal Drop |
|--------------|----------------------|--------|--------------|
| full_model    | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] | 90.0% |      +90.0% |
| mid_top8      | [12, 13, 14, 15, 16, 17, 18, 19] | 70.0% |      +70.0% |
| last_12       | [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] | 60.0% |      +60.0% |
| top_half      | [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] | 60.0% |      +60.0% |
| last_8        | [16, 17, 18, 19, 20, 21, 22, 23] |  0.0% |       +0.0% |
| last_6        | [18, 19, 20, 21, 22, 23] |  0.0% |       +0.0% |
| last_4        | [20, 21, 22, 23]     |  0.0% |       +0.0% |

**Most effective group**: `full_model` — ASR=90.0%, refusal_drop=+90.0%
Active layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

### Critical Refusal Layers (refusal drop > 5%)

- **Layer 9** (LINEAR_ATTN): refusal drop = +70.0%
- **Layer 10** (LINEAR_ATTN): refusal drop = +70.0%
- **Layer 6** (LINEAR_ATTN): refusal drop = +50.0%
- **Layer 8** (LINEAR_ATTN): refusal drop = +40.0%
- **Layer 11** (FULL_ATTN): refusal drop = +30.0%
- **Layer 5** (LINEAR_ATTN): refusal drop = +20.0%
- **Layer 7** (FULL_ATTN): refusal drop = +20.0%
- **Layer 12** (LINEAR_ATTN): refusal drop = +20.0%
- **Layer 14** (LINEAR_ATTN): refusal drop = +20.0%
- **Layer 13** (LINEAR_ATTN): refusal drop = +10.0%
- **Layer 21** (LINEAR_ATTN): refusal drop = +10.0%
- **Layer 22** (LINEAR_ATTN): refusal drop = +10.0%
- **Layer 23** (FULL_ATTN): refusal drop = +10.0%

### Full Attention Layers (n=6)
Indices: [3, 7, 11, 15, 19, 23]

### Linear Attention Layers (n=18)

## Per-Layer Detailed Results

| Layer | Type       | ASR     | Refusal Drop |
|-------|------------|---------|--------------|
|     0 | LINEAR     |    0.0% |        +0.0% |
|     1 | LINEAR     |    0.0% |        +0.0% |
|     2 | LINEAR     |    0.0% |        +0.0% |
|     3 | FULL       |    0.0% |        +0.0% |
|     4 | LINEAR     |    0.0% |        +0.0% |
|     5 | LINEAR     |   20.0% |       +20.0% |
|     6 | LINEAR     |   50.0% |       +50.0% |
|     7 | FULL       |   20.0% |       +20.0% |
|     8 | LINEAR     |   40.0% |       +40.0% |
|     9 | LINEAR     |   70.0% |       +70.0% |
|    10 | LINEAR     |   70.0% |       +70.0% |
|    11 | FULL       |   30.0% |       +30.0% |
|    12 | LINEAR     |   20.0% |       +20.0% |
|    13 | LINEAR     |   10.0% |       +10.0% |
|    14 | LINEAR     |   20.0% |       +20.0% |
|    15 | FULL       |    0.0% |        +0.0% |
|    16 | LINEAR     |    0.0% |        +0.0% |
|    17 | LINEAR     |    0.0% |        +0.0% |
|    18 | LINEAR     |    0.0% |        +0.0% |
|    19 | FULL       |    0.0% |        +0.0% |
|    20 | LINEAR     |    0.0% |        +0.0% |
|    21 | LINEAR     |   10.0% |       +10.0% |
|    22 | LINEAR     |   10.0% |       +10.0% |
|    23 | FULL       |   10.0% |       +10.0% |

## Top Layers by Refusal Direction Score

| Rank | Layer | Score |
|------|-------|-------|
|    1 | 23    | 5.3008 |
|    2 | 22    | 4.3281 |
|    3 | 21    | 3.9691 |
|    4 | 20    | 3.5154 |
|    5 | 19    | 3.0873 |
|    6 | 18    | 2.8315 |
|    7 | 17    | 2.5836 |
|    8 | 16    | 2.3376 |
|    9 | 15    | 1.9201 |
|   10 | 14    | 1.6953 |

## Conclusions

- TBD: analyze which attention types dominate the refusal signal
- TBD: identify whether refusal is concentrated in later layers
- TBD: compare with Qwen3.5-27B findings (layers 48-63)