# Vigilancia de influenza aviar en mamíferos

Proyecto de Minería de Datos (IIC-2433, 2026-2).
Integrantes: Sara Kita · Aline Kneuer · Antonia Martínez · Rubén Ariztía · Alonso Matta

**Pregunta guía:** ¿qué patrones tienen los brotes de influenza aviar en mamíferos y qué características se asocian a que un evento sea letal?

## Datos
- **AVIAN-MAMM**, archivo `latest-outbreaks-mammals.csv` (379 eventos, 48 columnas, marzo 2021 – abril 2024).
- Fuente: https://github.com/fbranda/avian-mamm · primaria: WAHIS (OMSA) · licencia **CC BY-SA 4.0**.
- Citar: Branda, F. et al. (2025). AVIAN-MAMM. *BMC Research Notes, 18*, 95.

## Estructura
```
proyecto/
├─ data/
│  ├─ raw/         CSV original (no se edita a mano)
│  ├─ processed/   avian_clean.csv, limpieza_log.txt, baseline_resultados.csv
│  └─ external/    contornos de países, solo para el mapa del EDA
├─ src/
│  ├─ 01_limpieza.py   limpieza y construcción de la etiqueta
│  ├─ 02_eda.py        análisis exploratorio (genera figs/)
│  └─ 03_baseline.py   baseline (clase mayoritaria) vs K-NN
├─ README.md
├─ requirements.txt
└─ run_all.sh
```

## Cómo reproducir
```bash
pip install -r requirements.txt
bash run_all.sh
```

## Decisiones de limpieza (detalle en `data/processed/limpieza_log.txt`)
- Longitud como texto con coma decimal → número. Una latitud sin punto decimal (42841 → 42,841).
- Nombres de especies duplicados unificados (37 → 35) y agrupados.
- `control_measures` (texto libre) → 10 variables binarias.
- Etiqueta `lethal`: 1 = death, 0 = positive, vacío = not certified (16, se analizan aparte).
- `is_farmed_wild_species`: 33 eventos de Finlandia en granjas registrados como "Wild".
- `total_deaths` y `total_killed_and_disposed_of` **no** deben usarse como predictores (fuga de información).
- Los faltantes de casos y muertes no se imputan aquí: imputar dentro de cada fold al modelar.

## Baseline (5 folds estratificados agrupados por `label_id`, 10 repeticiones)
| Modelo | Accuracy | F1 macro | F1 positivos |
|---|---|---|---|
| Baseline (siempre letal) | 0,91 | 0,48 | 0,00 |
| K-NN (k=5) | 0,90 | 0,62 | 0,30 |

## Próximos pasos
Clustering (DBSCAN/HDBSCAN, K-Means, GMM), reglas de asociación, red bayesiana y demo en vivo.
