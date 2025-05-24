import pandas as pd

# Load your cleaned data
df = pd.read_csv('players_cleaned.csv')

# ----------------------------
# Normalize Age
df['Age_norm'] = df['Age'] / 40

# Normalize Market Value
df['MarketValue_norm'] = df['MarketValue'] / 222_000_000

# Normalize Transfers
df['Transfers_norm'] = df['Transfer History Count'] / 15

# ----------------------------
# Nationality normalization based on continent
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

# ----------------------------
# Input Score = sum of normalized components
df['InputScore'] = (
    df['Age_norm'] +
    df['MarketValue_norm'] +
    df['Transfers_norm'] +
    df['Nationality_norm'] +
    df['ClubTier_norm']
).round(3)

# ----------------------------
# Sustainability Score = ASR / InputScore
df['SustainabilityScore'] = (df['ASR'] / df['InputScore']).round(3)

# ----------------------------
# Save results
df.to_csv('data_with_all_scores.csv', index=False)
print("✅ All normalizations and scores computed. Saved to 'data_with_all_scores.csv'")
