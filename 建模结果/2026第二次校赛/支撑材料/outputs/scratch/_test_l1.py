import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import importlib.util, pickle

spec = importlib.util.spec_from_file_location("S2model", "S2-model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open(m.DATA_IN, "rb"))
dd = p["per_disease"]["CRC"]
Xs = StandardScaler().fit_transform(dd["X_clr"])
y = dd["y"]

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    a = LogisticRegression(solver="liblinear", C=0.1, max_iter=2000, random_state=0, l1_ratio=1.0).fit(Xs, y).coef_[0]
    print("liblinear l1_ratio=1.0 (no penalty): nonzero", int((np.abs(a) > 1e-8).sum()), "| warnings:", len(w))
    for x in w:
        print("  WARN:", str(x.message)[:100])
