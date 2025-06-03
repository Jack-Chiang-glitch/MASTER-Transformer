# MASTER-Transformer

```txt
在這次研究中，比較了四種不同的模型不同的表現，
MASTER-Transformer, MASTER-Transformer(flash attention)
XGBoost Regressor, XGBoost PCA Regressor
使用了186個因子，使用XGBoost模型有用了各種特徵篩選方式
例如:檢查Multicolinearity, F regression score, XGBoost Importance, LASSO, ElasticNet等等

詳盡資訊(回測結果、統計檢定、Alphalnes評估統計IC值、資料集、特徵篩選、TimeSeries Cross Validation、模型比較)
```

## MASTER-Transformer（深度學習）與 XGBoost Regressor（傳統機器學習）

兩者的詳細比較請參閱：
**模型比較報告書 XGBoost（PCA）.pdf**

## MASTER-Transformer 與 MASTER-Transformer(Flash attention)
兩者的詳細比較請參閱：
**模型比較報告書(MASTER-Transformer v.s. flash attn).pdf**



<br><br>
<br>


# MASTER-Transformer架構

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
