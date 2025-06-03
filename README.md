# MASTER-Transformer
```txt
                         ┌────────────────────────────┐
                         │   Market Status Vector mτ  │
                         │ (Index Price & Volume etc.)│
                         └────────────┬───────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────────┐
                      │        1. Market-Guided Gating     │
                      │ Rescale xu,t ⇒ x̃u,t using α(mτ)    │
                      └────────────┬───────────────────────┘
                                   │
                                   ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│   Stock 1: x̃1,1…τ   │    │   Stock 2: x̃2,1…τ   │    │   Stock M: x̃M,1…τ   │
└────────┬───────────┘    └────────┬───────────┘    └────────┬───────────┘
         ▼                        ▼                        ▼
  ┌────────────┐          ┌────────────┐          ┌────────────┐
  │ Intra-Stock│          │ Intra-Stock│   ...    │ Intra-Stock│
  │ Aggregation│          │ Aggregation│          │ Aggregation│
  └─────┬──────┘          └─────┬──────┘          └─────┬──────┘
        ▼                      ▼                         ▼
     h1,1…τ                 h2,1…τ                  hM,1…τ   (local embeddings)
        └──────┬──────────────┴─────────────┬─────────────┘
               ▼                            ▼
       ┌────────────────────────────────────────────┐
       │      3. Inter-Stock Aggregation            │
       │ At each time step t, apply attention       │
       │ on all stocks’ h*,t ⇒ z*,t (temporal emb.) │
       └──────────────┬─────────────────────────────┘
                      ▼
             z1,1…τ, z2,1…τ, ..., zM,1…τ
                      ▼
       ┌──────────────────────────────────────┐
       │     4. Temporal Aggregation          │
       │ For each stock u, use zu,τ to attend │
       │ over zu,1…τ ⇒ final embedding eu     │
       └──────────────┬───────────────────────┘
                      ▼
       ┌────────────────────────────────────┐
       │       5. Prediction Layer           │
       │  Apply linear layer to eu ⇒ r̂u     │
       └────────────────────────────────────┘
```
