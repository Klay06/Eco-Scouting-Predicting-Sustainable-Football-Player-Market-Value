import pandas as pd

# Load the data with scores
df = pd.read_csv('data_with_all_scores_scaled.csv')

# Sort by SustainabilityScore descending and get top 10
top_10_sustainable = df.sort_values(by='SustainabilityScore', ascending=False).head(100)

# Display the top 10 players with their sustainability scores
print("Top 10 players by Sustainability Score:")
print(top_10_sustainable[['Player Name', 'SustainabilityScore']])

# Optionally save the top 10 to a new CSV file
top_10_sustainable.to_csv('top_100_sustainable_players.csv', index=False)
print("\n✅ Top 10 sustainable players saved to 'top_10_sustainable_players.csv'")