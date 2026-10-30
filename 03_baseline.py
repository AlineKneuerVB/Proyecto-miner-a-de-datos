# Baseline (clase mayoritaria) vs K-NN para predecir lethal (death vs positive). Excluye 'not certified'.
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT/"data/processed/avian_clean.csv")
d = d[d.lethal.notna()].reset_index(drop=True)
y = d.lethal.astype(int).values
cat = ["species_group", "typology_adj", "epidemiological_unit"]
num = ["log_cases"]
def model(kind):
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
    clf = DummyClassifier(strategy="most_frequent") if kind=="base" else KNeighborsClassifier(n_neighbors=5, weights="distance")
    return make_pipeline(pre, clf)
rows=[]
for seed in range(10):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for kind in ["base","knn"]:
        pred = np.zeros(len(d), int)
        for tr, te in cv.split(d, y, groups=d.label_id):
            m = model(kind).fit(d.iloc[tr], y[tr]); pred[te] = m.predict(d.iloc[te])
        rows.append(dict(model=kind, seed=seed, accuracy=accuracy_score(y,pred),
            f1_macro=f1_score(y,pred,average="macro",zero_division=0),
            f1_positive=f1_score(y,pred,pos_label=0,zero_division=0),
            recall_positive=recall_score(y,pred,pos_label=0,zero_division=0),
            precision_positive=precision_score(y,pred,pos_label=0,zero_division=0)))
r = pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns="seed").round(3)
print(r); r.to_csv(ROOT/"data/processed/baseline_resultados.csv")
