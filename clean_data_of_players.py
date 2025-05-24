import pandas as pd
import numpy as np

# 1. Load your CSV file
df = pd.read_csv('players_full_data.csv')

# 2. Remove players without valid ASR value
if 'ASR' in df.columns:
    df = df[~df['ASR'].astype(str).str.strip().isin(['', '-', 'N/A', 'nan'])]
else:
    raise ValueError("❌ The dataset does not contain an 'ASR' column.")

# 3. Extract and clean Age
df['Age_raw'] = df['Age'].str.extract(r'(\d+)').astype(float)
median_age = df['Age_raw'].median(skipna=True)
df['Age'] = df['Age_raw'].fillna(median_age).astype(int)

# 4. Parse Market Value strings into numeric euros
def parse_market_value(val):
    if isinstance(val, str):
        s = val.strip().replace(',', '')
        if s.endswith('K €'):
            return float(s[:-3]) * 1_000
        if s.endswith('M €'):
            return float(s[:-3]) * 1_000_000
    return np.nan

df['MarketValue_raw'] = df['Market Value'].map(parse_market_value)

# 5. Impute missing market values with the median (fallback to 0 if all are missing)
median_mv = df['MarketValue_raw'].median(skipna=True)
if np.isnan(median_mv):
    median_mv = 0.0
df['MarketValue'] = df['MarketValue_raw'].fillna(median_mv).round().astype(int)

# 6. Flag which market values were imputed
df['missing_market_value'] = df['MarketValue_raw'].isna()

# 7. Handle Transfer History Count: assume missing = 0
if 'Transfer History Count' in df.columns:
    df['Transfer History Count'] = df['Transfer History Count'].fillna(0).astype(int)

# 8. Drop helper columns
df = df.drop(columns=['Age_raw', 'Market Value', 'MarketValue_raw'])

# 9. Save the cleaned data
df.to_csv('players_cleaned.csv', index=False, encoding='utf-8')

print("✅ Cleaned data saved to 'players_cleaned.csv'")
