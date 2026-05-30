import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

# Display first rows
print("First 5 Rows")
print(df.head())

# Dataset Information
print("\nDataset Info")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

# Rename columns (remove extra spaces if present)
df.columns = df.columns.str.strip()

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Statistical Summary
print("\nSummary Statistics")
print(df.describe())

# -----------------------------
# Visualization 1
# Distribution of Unemployment
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Estimated Unemployment Rate (%)'],
             bins=20,
             kde=True)
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Visualization 2
# State-wise Average Unemployment
# -----------------------------
state_avg = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=state_avg.index,
            y=state_avg.values)

plt.xticks(rotation=90)
plt.title("Average Unemployment Rate by State")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -----------------------------
# Visualization 3
# Monthly Trend
# -----------------------------
monthly = df.groupby(df['Date'].dt.to_period('M'))[
    'Estimated Unemployment Rate (%)'
].mean()

monthly.index = monthly.index.astype(str)

plt.figure(figsize=(12,6))
plt.plot(monthly.index,
         monthly.values,
         marker='o')

plt.xticks(rotation=90)
plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# -----------------------------
# Visualization 4
# Labour Participation vs Unemployment
# -----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x='Estimated Labour Participation Rate (%)',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Labour Participation vs Unemployment")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

print("Analysis Completed Successfully!")
