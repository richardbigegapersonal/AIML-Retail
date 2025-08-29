# Retry with scikit-learn compatibility: OneHotEncoder(sparse=False)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
import networkx as nx

rng = np.random.default_rng(12)

# Data generation as before
n_stores, n_items, n_weeks = 40, 60, 12
stores = np.array([f"S{i:03d}" for i in range(n_stores)])
items  = np.array([f"I{j:03d}" for j in range(n_items)])
weeks  = np.arange(n_weeks)

alpha = -0.2
beta_price = -0.35
beta_promo = 0.55
beta_temp  = 0.08

store_eff = rng.normal(0.0, 0.35, size=n_stores)
item_eff  = rng.normal(0.0, 0.45, size=n_items)

rows = []
for s_idx, s in enumerate(stores):
    temp_anom = rng.normal(0, 1, size=n_weeks)
    for i_idx, it in enumerate(items):
        base_price = 4.5 + 0.4*rng.standard_normal() + 0.1*(i_idx % 5)
        promo_schedule = (rng.random(n_weeks) < 0.25).astype(int)
        price_idx = (base_price * (1 - 0.15*promo_schedule)) / base_price
        for w in weeks:
            x_price = price_idx[w]
            x_promo = promo_schedule[w]
            x_temp  = temp_anom[w]
            lam_log = (alpha + store_eff[s_idx] + item_eff[i_idx] +
                       beta_price*x_price + beta_promo*x_promo + beta_temp*x_temp)
            lam = np.exp(lam_log)
            y = rng.poisson(lam)
            rows.append((s, it, int(w), x_price, x_promo, x_temp, y))

df = pd.DataFrame(rows, columns=["store","item","week","price_idx","promo","temp_anom","units"])
train_df = df[df["week"] < n_weeks-2].copy()
valid_df = df[df["week"] >= n_weeks-2].copy()

feature_cols = ["price_idx","promo","temp_anom"]
cat_cols = ["store","item"]

def fit_and_eval(alpha_reg, include_fixed_effects=True, label=""):
    transformers = []
    transformers.append(("num", "passthrough", feature_cols))
    if include_fixed_effects:
        transformers.append(("cat", OneHotEncoder(sparse_output=False,handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
    model = PoissonRegressor(alpha=alpha_reg, max_iter=1000)
    pipe = Pipeline([("pre", pre), ("glm", model)])
    pipe.fit(train_df, train_df["units"])
    mu_hat = pipe.predict(valid_df)
    dev = mean_poisson_deviance(valid_df["units"], mu_hat)
    return pipe, dev, mu_hat

pipe_global, dev_global, mu_global = fit_and_eval(alpha_reg=1.0, include_fixed_effects=False, label="global")
pipe_nopool, dev_nopool, mu_nopool = fit_and_eval(alpha_reg=1e-8, include_fixed_effects=True, label="nopool")
pipe_partial, dev_partial, mu_partial = fit_and_eval(alpha_reg=1.0, include_fixed_effects=True, label="partial")

print("Mean Poisson deviance (hold-out, lower is better):")
print(f"  Complete pooling (features only): {dev_global:.4f}")
print(f"  No pooling   (store+item, ~0 L2): {dev_nopool:.4f}")
print(f"  Partial pool (store+item, L2=1): {dev_partial:.4f}")


