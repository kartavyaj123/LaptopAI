from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class LaptopFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Brand
        X["brand"] = (
            X["model"]
            .astype("string")
            .str.split()
            .str[0]
            .str.strip()
            .str.replace(r"[^a-zA-Z]", "", regex=True)
            .str.lower()
            .replace({
                "asus": "Asus",
                "lenovo": "Lenovo",
                "hp": "HP",
                "acer": "Acer",
                "msi": "MSI",
                "dell": "Dell",
                "apple": "Apple",
                "samsung": "Samsung",
                "microsoft": "Microsoft",
                "infinix": "Infinix",
                "alienware": "Alienware",
                "gigabyte": "Gigabyte",
                "ultimus": "Ultimus",
                "motorola": "Motorola",
                "thomson": "Thomson",
                "primebook": "Primebook",
                "colorful": "Colorful",
                "walker": "Walker",
                "air": "Air",
                "chuwi": "Chuwi"
            })
        )

        # RAM
        X["ram_gb"] = pd.to_numeric(
            X["ram"]
            .astype("string")
            .str.extract(
                r"(\d+(?:\.\d+)?)",
                expand=False
            ),
            errors="coerce"
        )

        # Storage
        storage_text = X["storage"].astype("string")

        storage_value = pd.to_numeric(
            storage_text.str.extract(
                r"(\d+(?:\.\d+)?)",
                expand=False
            ),
            errors="coerce"
        )

        storage_unit = (
            storage_text
            .str.extract(
                r"\d+(?:\.\d+)?\s*(GB|TB)",
                expand=False
            )
            .fillna("")
            .str.upper()
        )

        X["storage_gb"] = storage_value

        X.loc[storage_unit == "TB", "storage_gb"] = (
            storage_value[storage_unit == "TB"] * 1024
        )

        # Display size
        X["display_inches"] = pd.to_numeric(
            X["display"]
            .astype("string")
            .str.extract(
                r"(\d+(?:\.\d+)?)\s*inches",
                expand=False
            ),
            errors="coerce"
        )

        # Resolution
        resolution_values = X["display"].astype("string").str.extract(
            r"(\d+)\s*x\s*(\d+)",
            expand=True
        )

        X["resolution_width"] = pd.to_numeric(
            resolution_values[0],
            errors="coerce"
        )

        X["resolution_height"] = pd.to_numeric(
            resolution_values[1],
            errors="coerce"
        )

        # PPI
        X["ppi"] = (
            np.sqrt(
                X["resolution_width"] ** 2 +
                X["resolution_height"] ** 2
            )
            / X["display_inches"]
        )

        # CPU cores
        X["cpu_cores"] = pd.to_numeric(
            X["cores"]
            .astype("string")
            .str.extract(
                r"(\d+)\s*Cores?",
                expand=False
            ),
            errors="coerce"
        )

        # CPU threads
        X["cpu_threads"] = pd.to_numeric(
            X["cores"]
            .astype("string")
            .str.extract(
                r"(\d+)\s*Threads?",
                expand=False
            ),
            errors="coerce"
        )

        # Remove original columns
        X.drop(
            columns=[
                "model",
                "cores",
                "ram",
                "storage",
                "display"
            ],
            inplace=True
        )

        return X