import csv

import requests


def add_row_to_csv(csv_file, row_data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


def common_keys_dict(dict1, dict2):
    common_dict = {}
    for key in dict1:
        if key in dict2:
            common_dict[key] = dict1[key]
    return common_dict


token = "pk_4a9546b33fc6433db703cf2ca47b7dff"
symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={token}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()
for symbol_data in symbols_data:
    symbol = symbol_data['symbol']
    url = f"https://api.iex.cloud/v1/data/core/advanced_stats/{symbol}?token={token}"

    response = requests.get(url).json()
    for item in response:
        data1 = {}

        for key, value in item.items():
            if value is not None:
                data1[key] = value
            else:
                data1[key] = 1

        data2 = {
            'beta': None, 'totalCash': None, 'currentDebt': None, 'revenue': None,
            'grossProfit': None,
            'totalRevenue': None, 'EBITDA': None, 'revenuePerShare': None, 'revenuePerEmployee': None,
            'debtToEquity': None,
            'profitMargin': None, 'enterpriseValue': None, 'enterpriseValueToRevenue': None, 'priceToSales': None,
            'priceToBook': None, 'forwardPERatio': None, 'pegRatio': None, 'peHigh': None, 'peLow': None,
            'putCallRatio': None, 'marketcap': None, 'week52high': None,
            'week52low': None, 'week52highSplitAdjustOnly': None,
            'week52lowSplitAdjustOnly': None, 'week52change': None,
            'sharesOutstanding': None, 'float': None, 'avg10Volume': None, 'avg30Volume': None, 'day200MovingAvg': None,
            'day50MovingAvg': None, 'employees': None, 'ttmEPS': None, 'ttmDividendRate': None, 'dividendYield': None,
            'peRatio': None, 'maxChangePercent': None,
            'year5ChangePercent': None, 'year2ChangePercent': None, 'year1ChangePercent': None,
            'ytdChangePercent': None,
            'month6ChangePercent': None,
            'month3ChangePercent': None, 'month1ChangePercent': None, 'day30ChangePercent': None,
            'day5ChangePercent': None
        }

        data = common_keys_dict(data1, data2)
        print(data)
        keys = list(data.keys())
        r = []
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                key1 = keys[i]
                key2 = keys[j]
                try:
                    ratio_value = data[key1] / data[key2]
                except ZeroDivisionError:
                    ratio_value = 0
                r.append(ratio_value)

        add_row_to_csv("ratios.csv", r)
        print("Done")
