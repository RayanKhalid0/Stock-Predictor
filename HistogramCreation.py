import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

figures = []
symbol = 'AAPL'

for t in range(0, 10):
    start_date = '2020-01-01'
    end_date = '2024-05-29'

    data = yf.download(symbol, start=start_date, end=end_date)

    data = data['Close'][-(51 * (t + 1)):-(51 * t) or None]

    if len(data) < 51:
        print(f"Not enough data only {len(data)}")
        continue


    prices = data.values
    print(prices)

    df = pd.DataFrame({'Day': range(1, 52), 'Price': prices})

    df['Price_Next'] = df['Price'].shift(-1)
    df = df.dropna()

    num_bins = 20
    price_min = df[['Price', 'Price_Next']].min().min()
    price_max = df[['Price', 'Price_Next']].max().max()

    bins = np.linspace(price_min, price_max, num_bins + 1)

    df['Price_Bin'] = pd.cut(df['Price'], bins=bins, labels=False, include_lowest=True)
    df['Price_Next_Bin'] = pd.cut(df['Price_Next'], bins=bins, labels=False, include_lowest=True)

    fig = px.density_heatmap(
        df,
        x='Price_Bin',
        y='Price_Next_Bin',
        nbinsx=num_bins,
        nbinsy=num_bins,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        title=f'2D Histogram No.{t + 1} for {symbol}',
        xaxis_title='Price on Day N',
        yaxis_title='Price on Day N+1',
        xaxis_nticks=num_bins,
        yaxis_nticks=num_bins
    )

    figures.append(fig)

    fig.write_image(f"2d_histogram_iteration_{t+1}.png")

