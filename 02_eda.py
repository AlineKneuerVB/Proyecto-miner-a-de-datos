import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT = Path(__file__).resolve().parents[1]; F = ROOT/"figs"; F.mkdir(exist_ok=True)
d = pd.read_csv(ROOT/"data/processed/avian_clean.csv")
RED, GRAY, BG = "#8F2105", "#9AA0A6", "#D7DBDE"
plt.rcParams.update({"font.family":"DejaVu Sans","font.weight":"bold","axes.facecolor":BG,"figure.facecolor":BG,
  "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,"font.size":13})
def save(n): plt.savefig(F/n, dpi=200, bbox_inches="tight", facecolor=BG); plt.close()

# 1 mapa
w = json.load(open(ROOT/"data/external/world.geo.json"))
fig, ax = plt.subplots(figsize=(8.5,4.4))
def poly(coords):
    ax.add_patch(Polygon(coords, closed=True, fc="#C2C7CC", ec=BG, lw=0.4))
for f in w["features"]:
    g=f["geometry"]
    if g["type"]=="Polygon": poly(g["coordinates"][0])
    elif g["type"]=="MultiPolygon":
        for p in g["coordinates"]: poly(p[0])
o = d[d.country_name!="Chile"]; c = d[d.country_name=="Chile"]
ax.scatter(o.longitude,o.latitude,s=16,c="#333",alpha=.55,lw=0,label="Otros países (345)")
ax.scatter(c.longitude,c.latitude,s=26,c=RED,lw=0,label="Chile (34)",zorder=5)
ax.set_xlim(-170,60); ax.set_ylim(-58,78); ax.set_aspect("equal"); ax.axis("off")
ax.legend(loc="lower left",frameon=False,fontsize=12); save("mapa.png")

# 2 desbalance
fig, ax = plt.subplots(figsize=(5.2,4.2))
v = d.outcome.value_counts().reindex(["death","positive","not certified"])
b = ax.bar(["Muerte o\nsacrificio","Positivo","Sin\ncertificar"], v.values, color=[RED,GRAY,GRAY], width=.6)
for r,x in zip(b,v.values): ax.text(r.get_x()+r.get_width()/2,x+6,str(x),ha="center",fontsize=18,fontweight="bold")
ax.set_yticks([]); ax.tick_params(length=0); save("desbalance.png")

# 3 letalidad por tipo (solo certificados)
cert = d[d.lethal.notna() & d.typology_adj.isin(["Wild","Domestic","Farmed"])]
g = cert.groupby("typology_adj").lethal.agg(["mean","sum","count"]).reindex(["Wild","Farmed","Domestic"])
fig, ax = plt.subplots(figsize=(5.6,4.2))
lab=["Silvestres","Granjas de\nFinlandia","Domésticos"]
b = ax.bar(lab, g["mean"]*100, color=[RED,"#B9644F",GRAY], width=.6)
for r,(m,s,n) in zip(b,g.values): ax.text(r.get_x()+r.get_width()/2,m*100+2,f"{m*100:.0f} %\n({int(s)} de {int(n)})",ha="center",fontsize=13,fontweight="bold")
ax.set_ylim(0,125); ax.set_yticks([]); ax.tick_params(length=0); save("letalidad.png"); print(g)

# 4 faltantes
cols = ["total_deaths","control_measures","total_cases","end_date_outbreak","total_killed_and_disposed_of"]
raw = pd.read_csv(ROOT/"data/raw/latest-outbreaks-mammals.csv")
cols_raw = ["total_susceptible","total_vaccinated","total_killed_and_disposed_of","total_deaths","control_measures","total_cases"]
m = (raw[cols_raw].isna().mean()*100).sort_values()
m.index=[{"total_susceptible":"susceptibles","total_vaccinated":"vacunados","total_killed_and_disposed_of":"sacrificados","total_deaths":"muertes","control_measures":"medidas control","total_cases":"casos"}[i] for i in m.index]
fig, ax = plt.subplots(figsize=(5.6,3.9))
ax.barh(m.index, m.values, color=[RED if x>80 else GRAY for x in m.values], height=.6)
for i,x in enumerate(m.values): ax.text(x+1.5,i,f"{x:.0f} %",va="center",fontsize=15,fontweight="bold")
ax.set_xlim(0,110); ax.set_xticks([]); ax.tick_params(length=0,labelsize=15); save("faltantes.png")

# 5 baseline
r = pd.read_csv(ROOT/"data/processed/baseline_resultados.csv",index_col=0)
fig, ax = plt.subplots(figsize=(5.4,3.8))
import numpy as np
x=np.arange(3); w_=.36
for i,(k,col,l) in enumerate([("base",GRAY,"Baseline (siempre letal)"),("knn",RED,"K-NN (k=5)")]):
    vals=[r.loc[k,"accuracy"],r.loc[k,"f1_macro"],r.loc[k,"f1_positive"]]
    bb=ax.bar(x+(i-.5)*w_,vals,w_,color=col,label=l)
    for rr,vv in zip(bb,vals): ax.text(rr.get_x()+rr.get_width()/2,vv+.02,f"{vv:.2f}",ha="center",fontsize=12,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Accuracy","F1 macro","F1 positivos"]); ax.set_yticks([]); ax.set_ylim(0,1.15)
ax.tick_params(length=0); ax.legend(frameon=False,fontsize=11,loc="upper right"); save("baseline.png")
