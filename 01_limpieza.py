# %% [markdown]
# # 01 – Limpieza de AVIAN-MAMM (latest-outbreaks-mammals.csv)
# Entrada: data/raw/latest-outbreaks-mammals.csv  |  Salida: data/processed/avian_clean.csv
# Cada decisión queda registrada en `log` y se imprime al final (sirve para el video y el informe).

# %%
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path("..")
RAW = ROOT / "data/raw/latest-outbreaks-mammals.csv"
OUT = ROOT / "data/processed/avian_clean.csv"
log = []
def note(msg): log.append(msg); print("•", msg)

df = pd.read_csv(RAW)
note(f"Carga: {df.shape[0]} filas x {df.shape[1]} columnas")

# %% [markdown]
# ## 1. Coordenadas (problema detectado: longitud viene como texto)
# 10 longitudes usan coma decimal y espacio duro (`'\xa0-79,5'`); 1 latitud está sin punto decimal (42841 → 42.841).

# %%
n_coma = df["longitude"].astype(str).str.contains(",").sum()
df["longitude"] = pd.to_numeric(
    df["longitude"].astype(str).str.replace("\xa0", "", regex=False).str.replace(",", ".", regex=False).str.strip(),
    errors="coerce")
note(f"longitude: {n_coma} valores con coma decimal/espacio duro convertidos a float")

bad_lat = df["latitude"].abs() > 90
for i in df.index[bad_lat]:
    old = df.at[i, "latitude"]
    df.at[i, "latitude"] = old / 1000          # 42841 -> 42.841 (Grant County, EE.UU.; coherente con su longitud)
    note(f"latitude fila {i}: {old} -> {old/1000} (punto decimal perdido; {df.at[i,'location']}, {df.at[i,'country_name']})")
assert df["latitude"].between(-90, 90).all() and df["longitude"].between(-180, 180).all()

# %% [markdown]
# ## 2. Fechas

# %%
for c in ["report_date", "start_date_outbreak", "end_date_outbreak"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df["start_month"] = df["start_date_outbreak"].dt.to_period("M").astype(str)
df["duration_days"] = (df["end_date_outbreak"] - df["start_date_outbreak"]).dt.days
note(f"Fechas parseadas. end_date_outbreak faltante en {df.end_date_outbreak.isna().mean():.0%} (brotes sin cierre): duration_days solo existe donde hay cierre")

# %% [markdown]
# ## 3. Categóricas: unificar especies, unidad epidemiológica y tipo de animal

# %%
SPECIES = {
    "Cats": "Cat", "Domestic cat": "Cat", "Dogs": "Dog",
    "Racoon (Northern raccoon)": "Raccoon", "Bovine": "Cattle",
    "American Black Bear (black bear)": "Black bear", "Brown bear (Grizzly Bear)": "Brown bear",
    "North American river otter": "River otter", "American marten": "Marten",
    "European Pine Marten": "Marten", "South-American sea lion": "South American sea lion",
    "South American Coati": "Coati", "Huillin": "Huillin (southern river otter)",
}
df["species_clean"] = df["species"].str.strip().replace(SPECIES)

GROUP = {
    "Red Fox": "Canid", "Arctic Fox": "Canid", "Coyote": "Canid", "Raccoon dog": "Canid", "Dog": "Canid",
    "Striped Skunk": "Mustelid/skunk", "American Mink": "Mustelid/skunk", "Fisher": "Mustelid/skunk",
    "European Polecat": "Mustelid/skunk", "Marten": "Mustelid/skunk", "Mustelidae": "Mustelid/skunk",
    "River otter": "Mustelid/skunk", "Marine otter": "Mustelid/skunk", "Huillin (southern river otter)": "Mustelid/skunk",
    "South American sea lion": "Marine mammal", "Harbor Seal": "Marine mammal", "Gray Seal": "Marine mammal",
    "Northern Fur Seal": "Marine mammal", "Bottlenose dolphin": "Marine mammal",
    "Puma": "Felid", "Bobcat": "Felid", "Cat": "Felid", "Tiger": "Felid", "Lion": "Felid", "Amur Leopard": "Felid",
    "Black bear": "Ursid", "Brown bear": "Ursid", "Polar bear": "Ursid",
    "Raccoon": "Other", "Cattle": "Other", "Virginia Opossum": "Other", "Coati": "Other",
    "Abert's squirrel": "Other", "Brown skua": "Other", "Kelp Gull": "Other",
}
df["species_group"] = df["species_clean"].map(GROUP).fillna("Other")
note(f"species: {df.species.nunique()} nombres -> {df.species_clean.nunique()} unificados; species_group con {df.species_group.nunique()} grupos")
note("species_group es una agrupación NUESTRA (criterio biológico); revisar. Ojo: 'Brown skua' y 'Kelp Gull' son aves, no mamíferos (error de la fuente): se marcan en is_bird")
df["is_bird"] = df["species_clean"].isin(["Brown skua", "Kelp Gull"])

df["epidemiological_unit"] = df["epidemiological_unit"].replace({"-": np.nan, "Not applicable": "Not applicable"})
df["epidemiological_unit"] = df["epidemiological_unit"].fillna("Unknown")
df["typology_of_animal"] = df["typology_of_animal"].fillna("Unknown")
note("epidemiological_unit '-' (2) y typology_of_animal NaN (1) -> 'Unknown' (no imputamos a mano)")

# Finlandia: eventos en granjas (zorros, visones, perros mapache) registrados como 'Wild' (33 eventos)
farmed = (df.country_name == "Finland") & (df.epidemiological_unit == "Farm")
df["is_farmed_wild_species"] = farmed
df["typology_adj"] = np.where(farmed, "Farmed", df["typology_of_animal"])
note(f"{farmed.sum()} eventos de Finlandia en granjas (zorros, visones, perros mapache) marcados como is_farmed_wild_species; typology_adj los separa de 'Wild' (decidir con el grupo cuál usar)")

# %% [markdown]
# ## 4. Medidas de control: texto libre -> banderas binarias

# %%
MEASURES = {
    "ctl_disinfection": r"Disinfection",
    "ctl_disposal": r"Official disposal",
    "ctl_surv_within": r"Surveillance within",
    "ctl_surv_outside": r"Surveillance outside",
    "ctl_wildlife_ctrl": r"Control of wildlife",
    "ctl_selective_kill": r"Selective killing",
    "ctl_quarantine": r"Quarantine",
    "ctl_stamping_out": r"Stamping out",
    "ctl_zoning": r"Zoning",
    "ctl_screening": r"Screening",
}
cm = df["control_measures"].fillna("")
for k, pat in MEASURES.items():
    df[k] = cm.str.contains(pat, case=False, regex=True).astype(int)
df["ctl_none"] = cm.str.contains("No control measure", case=False).astype(int)
df["ctl_unknown"] = df["control_measures"].isna().astype(int)
df["ctl_count"] = df[list(MEASURES)].sum(axis=1)
df.loc[df.ctl_unknown == 1, "ctl_count"] = np.nan
note(f"control_measures: {len(MEASURES)} banderas + ctl_none + ctl_unknown ({df.ctl_unknown.sum()} sin dato, NO se asumen 'sin medidas')")

# %% [markdown]
# ## 5. Numéricas y etiqueta
# `outcome` se deriva de los conteos → `total_deaths` / `total_killed_and_disposed_of` NO pueden ser predictores.

# %%
df["outcome"] = df["outcome"].str.strip().str.lower()
df["lethal"] = df["outcome"].map({"death": 1, "positive": 0})      # 'not certified' -> NaN (se analiza aparte)
df["certified"] = df["outcome"].ne("not certified")
note(f"lethal: {int((df.lethal==1).sum())} letal, {int((df.lethal==0).sum())} positivo, {int(df.lethal.isna().sum())} 'not certified' (NaN, excluidos del modelado)")

for c in ["total_cases", "total_deaths"]:
    df[f"{c}_missing"] = df[c].isna().astype(int)
df["log_cases"] = np.log1p(df["total_cases"])
note("total_cases/total_deaths: NaN se conservan + banderas *_missing; log_cases = log1p. Imputar (mediana) SOLO dentro de cada fold al modelar, no aquí")

# %% [markdown]
# ## 6. Columnas descartadas y duplicados

# %%
na = df.isna().mean()
drop_na = [c for c in df.columns if na[c] > 0.80 and c in
           ["total_susceptible", "total_vaccinated", "total_slaughtered_killed_for_commercial_use",
            "detailed_characterisation", "test_name_3", "laboratory_name_3", "first_result_date_3",
            "latest_result_date_3", "result_3"]]
ids_texto = ["url_event", "url_pdf", "report_number", "country_iso3_digit", "affected_population_description"]
df_model = df.drop(columns=drop_na + ids_texto)
note(f"Descartadas por >80% NA: {drop_na}")
note(f"Descartadas por ser URLs/ids/texto largo (se conservan en el CSV crudo): {ids_texto}")
note(f"Filas duplicadas exactas: {df.duplicated().sum()}. label_id repetido en {df.label_id.duplicated().sum()} filas: son eventos distintos del mismo brote (misma ficha, otra especie/ubicación) -> NO se eliminan")

# %%
df_model.to_csv(OUT, index=False)
note(f"Guardado {OUT.name}: {df_model.shape[0]} filas x {df_model.shape[1]} columnas")
(ROOT / "data/processed/limpieza_log.txt").write_text("\n".join(log), encoding="utf-8")
