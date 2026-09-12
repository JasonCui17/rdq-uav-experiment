# MMUAV paper reproduction status

## M0 — Public 9D Ordinary LSTM

- Input: `[B,20,9]` with `mean_xyz`, `std_xyz`, and `range_xyz`.
- Model: `LSTM(9,64,1) → out[:,-1,:] → Linear(64,2)`.
- Train dataset: 58,701 clusters, including 140 positives.
- Validation dataset: 12,036 clusters, including 46 positives.
- Official public checkpoint validation F1: 0.9318.
- Retrained best validation F1: 0.9890.
- Best validation-loss epoch: 19; early stopping epoch: 34.

This is the public-code 9D ordinary-LSTM baseline. It is not the paper's
unreleased 7D Attention-LSTM implementation.

## Next controlled experiment

M1 uses the same 9D dataset, split, augmentation, LSTM dimensions, classifier,
loss, optimizer, learning rate, batch size, seed, and early-stopping rule. Its
only model change is scalar attention over all 20 LSTM hidden states instead of
selecting the final hidden state.
