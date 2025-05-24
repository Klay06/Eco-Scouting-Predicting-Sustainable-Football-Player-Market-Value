import os
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Determine base directory (script location or cwd)
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()

# 2. File paths
input_path = base_dir / 'players_full_data.csv'
output_path = base_dir / 'players_cleaned.csv'

# 3. Verify input file exists
if not input_path.is_file():
    available = [f.name for f in base_dir.iterdir() if f.is_file() and f.suffix == '.csv']
    raise FileNotFoundError(
        f"Could not find 'players_full_data.csv' in {base_dir}\n"
        f"CSV files present:\n  " + "\n  ".join(available)
    )

# 4. Load data
df = pd.read_csv(input_path)

# 5. Extract and clean Age
#    - Extract digits to a float column
#    - Fill NaN with median age
#    - Convert to int
df['Age_raw'] = df['Age'].str.extract(r'(\d+)').astype(float)
median_age = df['Age_raw'].median(skipna=True)
df['Age'] = df['Age_raw'].fillna(median_age).astype(int)

# 6. Parse Market Value strings into numeric euros
def parse_market_value(val):
    if isinstance(val, str):
        s = val.strip().replace(',', '')
        if s.endswith('K €'):
            return float(s[:-3]) * 1_000
        if s.endswith('M €'):
            return float(s[:-3]) * 1_000_000
    return np.nan

df['MarketValue_raw'] = df['Market Value'].map(parse_market_value)

# 7. Impute missing market values with median (fallback to 0)
median_mv = df['MarketValue_raw'].median(skipna=True)
if np.isnan(median_mv):
    median_mv = 0.0
df['MarketValue'] = df['MarketValue_raw'].fillna(median_mv).round().astype(int)

# 8. Flag imputed market values
df['missing_market_value'] = df['MarketValue_raw'].isna()

# 9. Handle transfers: missing → 0, and flag
if 'Transfer History Count' in df.columns:
    df['missing_transfers'] = df['Transfer History Count'].isna()
    df['Transfer History Count'] = df['Transfer History Count'].fillna(0).astype(int)

# 10. Drop helper columns
drop_cols = ['Market Value', 'MarketValue_raw', 'Age_raw']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 11. Save cleaned data
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"✅ Data cleaned and saved to '{output_path}' ({len(df)} rows).")
