# Freight Rate Prediction Challenge

## Clone the project

```bash 
git clone https://github.com/Uzairkheshgi/freight-rate-prediction
```
## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the scrip

```bash
python train.py
```

Trains on `data/train_test.csv`, prints the holdout score, then writes
`validation_predictions.csv` and fills the `predicted_rate` column of
`data/december_chart_inputs.csv`. Takes about a minute. 

## Run score.py
To produce the chart run below command.

```bash
python score.py --predictions validation_predictions.csv --december-predictions 
```

## Layout

| Path | What it is |
|---|---|
| `eda/eda.ipynb` | Exploration, data-quality findings and the feature benchmarks.|
| `data_cleaning.py` | Loading, cleaning, feature building, the time split. Imported, not run for its side effects. |
| `train.py` | Trains the model and writes both prediction files. |
| `score.py` | To produce the chart. |
