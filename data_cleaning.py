"""Load and clean the freight data for training."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_START = pd.Timestamp("2025-01-01")
IQR_MULTIPLIER = 1.5

EQUIPMENT_DTYPE = pd.CategoricalDtype(["Dry Van", "Flatbed", "Reefer"])

FEATURES = [
    "log_distance",
    "distance",
    "weight",
    "equipment",
    "day_of_week",
    "day_of_year",
    "days_elapsed",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]


def load(filename):
    """Read one of the CSVs in data/."""
    return pd.read_csv(DATA_DIR / filename, parse_dates=["date"])


def clean(df):
    """Fix the negative weight signs. Missing weights stay NaN for LightGBM."""
    cleaned = df.copy()
    cleaned["weight"] = cleaned.weight.abs()
    return cleaned


def city_coords(df):
    """Map each city to its coordinate from the pickup and delivery columns."""
    columns = ["city", "lat", "lon"]
    pickups = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(columns, axis=1)
    deliveries = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
        columns, axis=1
    )
    return pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city")


def add_coords(df, coords):
    """Fill in coordinates for a frame that arrived without them, in memory only."""
    located = df.copy()
    for role in ("pickup", "delivery"):
        if f"{role}_lat" not in located.columns:
            located[f"{role}_lat"] = located[role].map(coords.lat)
            located[f"{role}_lon"] = located[role].map(coords.lon)
    return located


def add_features(df):
    """Attach the model inputs. Run clean() first."""
    featured = df.copy()
    featured["log_distance"] = np.log(featured.distance)
    featured["day_of_week"] = featured.date.dt.dayofweek
    featured["day_of_year"] = featured.date.dt.dayofyear
    featured["days_elapsed"] = (featured.date - TRAIN_START).dt.days
    featured["equipment"] = featured.equipment.astype(EQUIPMENT_DTYPE)
    if "posted_rate" in featured.columns:
        featured["rpm"] = featured.posted_rate / featured.distance
    return featured


def prepare(df, coords=None):
    """Clean and build features in one call. Pass coords to fill missing lat/lon."""
    prepared = clean(df)
    if coords is not None:
        prepared = add_coords(prepared, coords)
    return add_features(prepared)


def time_split(df, cutoff):
    """Split into the rows before the cutoff and the rows from it onward."""
    cutoff = pd.Timestamp(cutoff)
    return df[df.date < cutoff].copy(), df[df.date >= cutoff].copy()


def rpm_bounds(df, multiplier=IQR_MULTIPLIER):
    """Tukey outlier fences on rate per mile, measured on the rows being clipped."""
    q1, q3 = df.rpm.quantile([0.25, 0.75])
    reach = multiplier * (q3 - q1)
    return q1 - reach, q3 + reach


def winsorize_target(df, bounds=None):
    """Clip posted_rate into the rpm fences. Fit folds only, never scored data."""
    low, high = rpm_bounds(df) if bounds is None else bounds
    clipped = df.copy()
    clipped["posted_rate"] = clipped.posted_rate.clip(
        clipped.distance * low, clipped.distance * high
    )
    return clipped


if __name__ == "__main__":
    train = prepare(load("train_test.csv"))
    fit, holdout = time_split(train, "2025-09-01")
    low, high = rpm_bounds(fit)
    print(f"train {len(train):,} rows | fit {len(fit):,} | holdout {len(holdout):,}")
    print(f"negative weights remaining: {(train.weight < 0).sum()}")
    print(f"rpm fences on the fit fold: {low:.2f} - {high:.2f} $/mile")
