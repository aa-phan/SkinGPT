# SkinGPT

A Streamlit-based demo app for skin condition analysis and product matching.

This repository contains a small computer-vision model (ResNet-18), preprocessing tools, and a product matching pipeline that uses ingredient-level efficacy vectors to recommend skincare routines.

Quick contents

- `app.py` — Streamlit app (upload an image, run the model, get product recommendations)
- `product_db/` — Utilities for cleaning product data, scoring ingredients, and matching products
  - `data_cleaner.py` — cleans raw product CSV and exports `dataset/skincare_db.csv`
  - `score_engine.py` — converts LLM ingredient classifications into `efficacy_vector` and outputs `dataset/final_sephora_database.csv`
  - `match_engine.py` — `SkincareMatchingEngine` which consumes a user condition vector and the final DB to assemble a routine
- `dataset/` — data artifacts (not all are committed; see notes below)

Setup

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Running the app locally

```powershell
streamlit run app.py
```
