# V1 to V2-base Equivalence Report

- Result: **V2_BASE_EQUIVALENCE = PASS**
- Seed/device/dtype: 42 / cpu / float32
- Strict state load: `<All keys matched successfully>`
- V1 parameters: 1,047,722
- V2 parameters: 1,047,722
- V1 config SHA-256: `ea35eed6274464441a605c5a3cde5a1579a5b96975adb1c7e8a7d0b9757447ea`
- V2 config SHA-256: `df5733b2b9313b49b03f1ed3a55090c42569530071fe9bf347d2862259bce530`
- Only config difference: `model.name: lidar_uav_v1 -> lidar_uav_v2`
- Equal after excluding `model.name`: True

## Real input

- Sample: `seq0001_valref_000001` (`seq0001`, index 1)
- t0: 1706255621.6135716
- Events: 1
- Points: 4111
- L0/L1/L2 tokens: [2566, 1487, 724]
- Logits shape: [2566]
- Residual/predicted XYZ shape: [2566, 3]
- Fine feature shape: [2566, 128]

## Numerical comparison

| Value | Max absolute difference |
|---|---:|
| Voxel centers | 0.0 |
| Objectness logits | 0.0 |
| XYZ residual | 0.0 |
| Predicted XYZ | 0.0 |
| 128D fine features | 0.0 |
| Total loss | 0.0 |

V1/V2 loss: 1.3139147758483887 / 1.3139147758483887. Positive, ignore, and negative masks are all exactly equal.

Raw candidates: 20 vs 20; score diff 0.0; XYZ diff 0.0; source IDs equal True.

NMS candidates: 20 vs 20; score diff 0.0; XYZ diff 0.0; source IDs equal True.
