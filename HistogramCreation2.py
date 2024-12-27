import pandas as pd
import numpy as np
import plotly.graph_objects as go

np.random.seed(0)
prices = np.random.uniform(low=90, high=200, size=51)
df = pd.DataFrame({'Day': range(1, 52), 'Price': prices})

df['Price_Next'] = df['Price'].shift(-1)
df = df.dropna()  # Remove the last row with NaN

print(df)

num_bins = 20
price_min = df[['Price', 'Price_Next']].min().min()
price_max = df[['Price', 'Price_Next']].max().max()

bins = np.linspace(price_min, price_max, num_bins + 1)
bin_labels = range(num_bins)

df['Price_Bin'] = pd.cut(df['Price'], bins=bins, labels=bin_labels, include_lowest=True)
df['Price_Next_Bin'] = pd.cut(df['Price_Next'], bins=bins, labels=bin_labels, include_lowest=True)

matrix = np.zeros((num_bins, num_bins), dtype=int)


for _, row in df.iterrows():
    i = int(row['Price_Bin'])
    j = int(row['Price_Next_Bin'])
    matrix[i, j] += 1

print(matrix)

fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=bins[:-1],
    y=bins[:-1],
    colorscale='Viridis'
))

fig.update_layout(
    title='2-D Histogram of Stock Prices',
    xaxis_title='Price on Day N',
    yaxis_title='Price on Day N+1'
)

import plotly.express as px