# Training

## The loss

Total loss = `task` + `α(t) · diversity` + `β(t) · similarity` where `α(t), β(t)` come from a schedule.

- **task** — cross-entropy over the MoE final logits.
- **diversity** — mean pairwise cosine similarity of L2-normalized expert features. Lower is more diverse. (Sign convention: the value is added *positively* to the total loss; if your paper convention is to *encourage* diversity, negate it or swap the schedule sign.)
- **similarity** — mean `KL(softmax(moe) || softmax(expert))` over experts, pushing each expert to agree with the gated ensemble.

## Schedules

Provided in `mme.losses.dynamic_balance`:

```python
from mme.losses import linear_schedule, cosine_schedule, step_schedule

linear_schedule(alpha_start=0.0, alpha_end=0.1, beta_start=0.0, beta_end=0.05)
cosine_schedule(alpha_start=0.0, alpha_end=0.1, beta_start=0.0, beta_end=0.05)
step_schedule([(0, 0.0, 0.0), (500, 0.05, 0.02), (2000, 0.1, 0.05)])
```

Or supply your own callable `(step, total_steps) -> (alpha, beta)`.

## Trainer

`mme.training.trainer.Trainer` is a small reference loop. Look at
`examples/train_toy.py` for a full example. If you want distributed training,
fork the Trainer or use `MMEModel.torch_parameters()` inside your own loop.

## Checkpointing

Set `ckpt_dir` on `Trainer` to save per-epoch state dicts of every torch expert and the gate.

## Watching gate specialization

The trainer prints the mean gate weights per epoch. In the toy example you should see them drift away from a uniform `1/N` distribution as experts specialize.
