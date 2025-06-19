import pandas as pd

# Load your cleaned data
df = pd.read_csv('players_cleaned.csv')

# ----------------------------
# Normalize Age (original 0-1 scale)
df['Age_norm'] = df['Age'] / 40
# Scale Age_norm to [0.1, 1]
df['Age_norm'] = 0.1 + 0.9 * df['Age_norm']

# Normalize Market Value (original 0-1 scale)
df['MarketValue_norm'] = df['MarketValue'] / 222_000_000
# Scale MarketValue_norm to [0.1, 1]
df['MarketValue_norm'] = 0.1 + 0.9 * df['MarketValue_norm']

# Normalize Transfers (original 0-1 scale)
df['Transfers_norm'] = df['Transfer History Count'] / 15
# Scale Transfers_norm to [0.1, 1]
df['Transfers_norm'] = 0.1 + 0.9 * df['Transfers_norm']

# ----------------------------
# Nationality normalization based on continent (already 0.5 or 1, scale accordingly)
european_countries = {
    'ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BEL', 'BIH', 'BGR', 'CHE', 'CYP',
    'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GEO', 'GRC',
    'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KAZ', 'LIE', 'LTU', 'LUX', 'LVA',
    'MCO', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU',
    'RUS', 'SMR', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'VAT'
}

def nationality_norm(nat):
    return 0.5 if nat in european_countries else 1.0

df['Nationality_norm'] = df['Nationality'].apply(nationality_norm)
# Scale Nationality_norm to [0.1, 1]
df['Nationality_norm'] = 0.1 + 0.9 * ((df['Nationality_norm'] - 0.5) / (1.0 - 0.5))

# ----------------------------
# Club Tier Mapping
tier_1_clubs = [
    'Manchester City', 'Paris Saint-Germain', 'Real Madrid', 'Barcelona',
    'Bayern Munich', 'Liverpool', 'Chelsea', 'Arsenal', 'Juventus', 'Inter Milan'
]

tier_2_clubs = [
    'Atalanta', 'Benfica', 'PSV Eindhoven', 'AS Roma', 'AC Milan', 'Sporting CP',
    'Sevilla', 'Borussia Dortmund', 'RB Leipzig', 'Napoli', 'Red Bull Salzburg'
]

def get_club_tier(club):
    if club in tier_1_clubs:
        return 1
    elif club in tier_2_clubs:
        return 2
    else:
        return 3  # Tier 3 for all other teams not in T1/T2

df['ClubTier'] = df['Team'].apply(get_club_tier)

# Normalize Club Tier (divide by 3 because 3 tiers now)
df['ClubTier_norm'] = df['ClubTier'] / 3
# Scale ClubTier_norm to [0.1, 1]
df['ClubTier_norm'] = 0.1 + 0.9 * df['ClubTier_norm']

# ----------------------------
# Define weights
weight_age = 0.2
weight_market = 0.04
weight_transfers = 0.32
weight_nationality = 0.28
weight_clubtier = 0.16

# Compute weighted InputScore
df['InputScore'] = (
    weight_age * df['Age_norm'] +
    weight_market * df['MarketValue_norm'] +
    weight_transfers * df['Transfers_norm'] +
    weight_nationality * df['Nationality_norm'] +
    weight_clubtier * df['ClubTier_norm']
).round(3)
# ----------------------------
# Sustainability Score = ASR / InputScore
df['SustainabilityScore'] = (df['ASR'] / df['InputScore']).round(3)

# ----------------------------
# Save results
df.to_csv('data_with_all_scores_scaled.csv', index=False)
print("✅ All values scaled to [0.1,1], scores computed, saved to 'data_with_all_scores_scaled.csv'")
