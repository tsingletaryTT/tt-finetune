# Real runs vs the planner (8-10 steps each, p300c one chip, 2026-11-23)

| Run | Result | Median step | Tokens/s |
|---|---|---|---|
| 7B auto (mixed BF8/BF4), seq 512 | OOM in first compiled step | | |
| 7B auto, seq 2048 | OOM in first compiled step | | |
| 7B bf4_frozen, seq 512 | OOM in first compiled step | | |
| 7B bf4_frozen, seq 384 | OOM in first compiled step | | |
| 7B bf4_frozen, seq 256 | OOM in first compiled step | | |
| 7B bf4_frozen, seq 128 | ok | 0.447 s | 286 |
| 1.5B auto, seq 2048 | ok | 3.716 s | 551 |
| 1.5B auto, seq 1024 | ok | 2.261 s | 453 |
