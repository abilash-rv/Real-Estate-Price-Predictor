
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq

# 1. File configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "housing data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CHUNK_SIZE = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Dataset path:", INPUT_FILE)
print("Dataset exists:", os.path.exists(INPUT_FILE))

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Housing dataset not found!")

# 2. Check dataset

df = pd.read_csv(INPUT_FILE)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATASET SHAPE")
print(df.shape)

del df

# 3. Initialize statistics

count = 0
mean = 0.0
M2 = 0.0
eda_sample = []

print("\nReading dataset in chunks...")

# 4. First pass: Calculate statistics

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

    chunk["SalePrice"] = pd.to_numeric(
        chunk["SalePrice"], errors="coerce"
    )

    prices = chunk["SalePrice"].dropna()
    prices = prices[prices > 0]

    n = len(prices)

    if n > 0:
        batch_mean = prices.mean()
        batch_M2 = ((prices - batch_mean) ** 2).sum()

        delta = batch_mean - mean
        new_count = count + n

        M2 += batch_M2 + delta**2 * count * n / new_count
        mean += delta * n / new_count
        count = new_count

        eda_sample.append(
            prices.sample(
                min(1000, len(prices)),
                random_state=42
            )
        )

if count == 0:
    raise ValueError("No valid house prices found!")

std = np.sqrt(M2 / count)

lower_limit = mean - 3 * std
upper_limit = mean + 3 * std

print("\nOUTLIER LIMITS")
print("Mean:", mean)
print("Standard Deviation:", std)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# 5. Exploratory Data Analysis

sample_prices = pd.concat(eda_sample)

plt.figure(figsize=(10, 5))

sns.histplot(
    sample_prices,
    bins=50,
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_distribution.png")
)

plt.show()

# 6. Second pass: Clean dataset

writer = None
total_removed = 0
total_saved = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

    numeric_cols = [
        "SalePrice",
        "YearBuilt",
        "YrSold",
        "GrLivArea"
    ]

    for col in numeric_cols:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        )

    chunk = chunk.dropna(subset=numeric_cols)

    chunk = chunk[
        (chunk["SalePrice"] > 0) &
        (chunk["YearBuilt"] > 0) &
        (chunk["YrSold"] >= chunk["YearBuilt"]) &
        (chunk["GrLivArea"] > 0)
    ].copy()

    # 7. Remove outliers

    before = len(chunk)

    chunk = chunk[
        (chunk["SalePrice"] >= lower_limit) &
        (chunk["SalePrice"] <= upper_limit)
    ].copy()

    total_removed += before - len(chunk)

    # 8. Feature Engineering

    # Feature 1: Home Age

    chunk["HomeAge"] = (
        chunk["YrSold"] - chunk["YearBuilt"]
    )

    # Feature 2: Price Per Square Foot

    chunk["PricePerSqFt"] = (
        chunk["SalePrice"] / chunk["GrLivArea"]
    )

    # Feature 3: Living Area Category

    chunk["LivingAreaCategory"] = pd.cut(
        chunk["GrLivArea"],
        bins=[0, 1000, 2000, 3000, np.inf],
        labels=["Small", "Medium", "Large", "Very Large"]
    )

    # 9. Save as Parquet

    table = pa.Table.from_pandas(
        chunk,
        preserve_index=False
    )

    if writer is None:
        writer = pq.ParquetWriter(
            os.path.join(
                OUTPUT_DIR,
                "cleaned_data.parquet"
            ),
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    total_saved += len(chunk)

# 10. Close Parquet writer

if writer is not None:
    writer.close()

print("\nPROCESSING COMPLETED")

print("Removed records:", total_removed)
print("Saved records:", total_saved)

print("Output folder:", OUTPUT_DIR)
print("Output: cleaned_data.parquet")
