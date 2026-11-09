"""Load public census data or produce explicitly labelled synthetic demo data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

TARGET = "MedHouseVal"
FEATURES = (
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
)


def make_synthetic_housing(n: int = 700, seed: int = 42) -> pd.DataFrame:
    """Generated data solely for offline smoke tests. Never report as census performance."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(32.5, 41.9, n)
    lon = rng.uniform(-124.2, -114.2, n)
    income = rng.gamma(shape=4.0, scale=1.0, size=n).clip(0.5, 15.0)
    age = rng.integers(1, 53, n).astype(float)
    rooms = rng.uniform(2, 8, n)
    bedrooms = rng.uniform(0.5, 2.2, n)
    population = rng.lognormal(mean=7.0, sigma=0.6, size=n)
    occupancy = rng.uniform(1.5, 6.0, n)
    coastal = np.exp(-((lon + 122) / 2.5) ** 2)
    target = (0.45 + 0.34 * income + 0.3 * coastal +
              0.07 * rooms - 0.04 * occupancy + rng.normal(0, 0.4, n))
    return pd.DataFrame({
        "MedInc": income, "HouseAge": age, "AveRooms": rooms,
        "AveBedrms": bedrooms, "Population": population, "AveOccup": occupancy,
        "Latitude": lat, "Longitude": lon, TARGET: np.clip(target, 0.15, 5.0),
    })


def load_housing(source: str = "california", synthetic_rows: int = 700) -> pd.DataFrame:
    if source == "synthetic":
        return make_synthetic_housing(n=synthetic_rows)
    if source != "california":
        raise ValueError("source must be either 'california' or 'synthetic'")
    try:
        bunch = fetch_california_housing(as_frame=True)
    except Exception as exc:
        raise RuntimeError(
            "California Housing download failed. It needs network access the first time. "
            "Run '--source synthetic' ONLY for an offline smoke test."
        ) from exc
    df = bunch.frame.copy()
    if TARGET not in df.columns:
        df[TARGET] = bunch.target
    return df


def location_groups(frame: pd.DataFrame, grid_degrees: float = 0.5) -> np.ndarray:
    """Geographic bins: hold out entire cells, not neighboring rows within the same cell."""
    lat = np.floor(frame["Latitude"].to_numpy() / grid_degrees).astype(int)
    lon = np.floor(frame["Longitude"].to_numpy() / grid_degrees).astype(int)
    return np.char.add(np.char.add(lat.astype(str), "_"), lon.astype(str))
