import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

import data_cleaning as dc

HOLDOUT_START = "2025-09-01"  # Sep-Oct: scored
EARLY_STOP_START = "2025-08-01"  # last month of the fit fold, picks the tree count
FINAL_STOP_START = "2025-10-01"

# Optimized via Optuna HPO to minimize Out-of-Time MAE
PARAMS = {
    "objective": "mae",
    "n_estimators": 3000,
    "learning_rate": 0.05152835868091922,
    "num_leaves": 57,
    "min_child_samples": 78,
    "colsample_bytree": 0.8057049105006229,
    "subsample": 0.7717204103206975,
    "subsample_freq": 1,
    "random_state": 0,
    "verbose": -1,
}


def fit_time_term(df):
    """Linear trend on the daily mean of log(rate per mile).

    Trees cannot extrapolate, so without a separate time term every December date lands in
    the same leaf and the chart flat-lines.
    """
    daily = np.log(df.rpm).groupby(df.days_elapsed).mean()
    days = daily.index.to_numpy()
    slope, _ = np.polyfit(days, daily.to_numpy(), 1)
    origin = days.mean()
    return lambda elapsed: slope * (elapsed - origin)


def residual_target(df, time_term):
    """log(rate) with the trend removed - what the trees actually learn."""
    return np.log(df.posted_rate) - time_term(df.days_elapsed)


def choose_n_trees(df, time_term, cutoff):
    """Early-stop on the last month of the fold being fitted, leaving the holdout untouched."""
    inner, stopping = dc.time_split(df, cutoff)
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        inner[dc.FEATURES],
        residual_target(inner, time_term),
        eval_X=stopping[dc.FEATURES],
        eval_y=residual_target(stopping, time_term),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.best_iteration_


def fit_model(df, time_term, n_trees):
    """Fit the features against the detrended target."""
    model = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": n_trees})
    model.fit(df[dc.FEATURES], residual_target(df, time_term))
    return model


def predict(model, df, time_term):
    """Add the trend back and return rates in dollars."""
    return np.exp(
        model.predict(df[dc.FEATURES]) + time_term(df.days_elapsed.to_numpy())
    )


def main():
    raw = dc.load("train_test.csv")
    coords = dc.city_coords(raw)
    train = dc.prepare(raw)

    # Fit Jan-Aug, score Sep-Oct. Only the fit fold is winsorised.
    fit, holdout = dc.time_split(train, HOLDOUT_START)
    fit = dc.winsorize_target(fit)
    trend = fit_time_term(fit)

    n_trees = choose_n_trees(fit, trend, EARLY_STOP_START)
    scored = predict(fit_model(fit, trend, n_trees), holdout, trend)

    baseline = fit.rpm.median() * holdout.distance
    print(f"\nfit {len(fit):,} rows -> holdout {len(holdout):,} rows, {n_trees} trees")
    print(f"baseline   MAE ${mean_absolute_error(holdout.posted_rate, baseline):,.2f}")
    print(
        f"model      MAE ${mean_absolute_error(holdout.posted_rate, scored):,.2f}   "
        f"MAPE {(np.abs(scored - holdout.posted_rate) / holdout.posted_rate).mean():.2%}"
    )

    # Refit on everything for December - Sep and Oct are the months closest to December.
    full = dc.winsorize_target(train)
    trend = fit_time_term(full)
    model = fit_model(full, trend, choose_n_trees(full, trend, FINAL_STOP_START))

    template = pd.read_csv(dc.DATA_DIR / "validation_predictions_template.csv")
    valid = dc.prepare(dc.load("validation.csv"))
    rates = pd.Series(predict(model, valid, trend), index=valid.load_id)
    template["predicted_rate"] = template.load_id.map(rates).round(2)
    assert template.predicted_rate.notna().all(), "every load_id must get a prediction"
    template.to_csv("validation_predictions.csv", index=False)

    # December has no coordinates, so they are filled in memory
    december = dc.load("december_chart_inputs.csv")
    december["predicted_rate"] = predict(
        model, dc.prepare(december, coords), trend
    ).round(2)
    december.to_csv(dc.DATA_DIR / "december_chart_inputs.csv", index=False)

    preds = december.predicted_rate
    print(f"\nvalidation {len(template):,} predictions -> validation_predictions.csv")
    print(
        f"december   {len(december)} predictions  ${preds.min():,.2f} - ${preds.max():,.2f} "
        f"(spread ${preds.max() - preds.min():,.2f})"
    )


if __name__ == "__main__":
    main()
