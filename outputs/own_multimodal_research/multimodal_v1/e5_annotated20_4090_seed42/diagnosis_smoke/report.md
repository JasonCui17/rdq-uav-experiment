# Multimodal E5 checkpoint diagnosis

## 1. Core 3D comparison

| Metric | best.ckpt | last.ckpt |
|---|---:|---:|
| Fusion Success@1m | 71.88% | 34.38% |
| Fusion Median Error | 0.8066 m | 36.9903 m |
| Radar pre-fusion Top1 Success@1m | 75.00% | 71.88% |
| Radar pre-fusion Top1 Median Error | 0.8092 m | 0.9410 m |

## 2. Final Top1 hypothesis type

| Type | best | last |
|---|---:|---:|
| R | 25 (78.12%) | 3 (9.38%) |
| RV | 6 (18.75%) | 11 (34.38%) |
| V | 1 (3.12%) | 18 (56.25%) |

## 3. Failure localization

### last.ckpt

- Radar Top1 已经在 1m 内，但最终 Fusion 失败：13
- 至少存在一个 1m 内 Radar 候选，但最终 Fusion 失败：13
- 最终选中的假设拥有 1m 内 Radar prior，但 Decoder refine 后失败：0

## 4. best -> last collapsed queries

- best 成功而 last 失败：13
- 其中 last 的 Radar Top1 仍在 1m 内：13
- 其中 last 至少仍有一个 Radar 候选在 1m 内：13
- 其中 selected Radar prior 正确，但最终 XYZ refine 后失败：0

### Collapsed query 的 last Top1 类型

```json
{
  "V": 13
}
```

## 5. Interpretation guide

- 如果 last 的 Radar pre-fusion Top1 也大幅崩溃：优先检查 T2/T3 解冻 Radar 后的参数更新。
- 如果 Radar pre-fusion 仍然准确，但 Fusion 失败：优先检查 hypothesis 排序 / Reliability Gate。
- 如果最终选择的 hypothesis 的 Radar prior 在 1m 内，但 refine 后 XYZ 出界：优先检查 Fusion Decoder / XYZ residual。
- 如果 V 类型大量成为 Top1 且错误率明显更高：重点检查 V hypothesis 的 3D 补全和跨类型 fused-score 校准。

> 注意：这里的 Radar pre-fusion 指 E5 内部经过当前 HCI/backbone 后的 Radar CandidateSet，不等同于独立 E0 baseline。
