import csv
import requests

def append(value, filename):
    with open(filename, "a") as file:
        file.write(f"{value}\n")


def totalCash(symbol):
    global token
    url = f"https://api.iex.cloud/v1/data/core/advanced_stats/{symbol}?token={token}"
    response2 = (requests.get(url)).json()
    print(response2)
    for item in response2:
        for key, value in item.items():
            if key == 'totalCash':
                return value


token = "pk_4a9546b33fc6433db703cf2ca47b7dff"

industries = [
    [0, 0, 'Manufacturing'],
    [0, 0, None],
    [0, 0, 'Finance and Insurance'],
    [0, 0, 'Educational Services'],
    [0, 0, 'Agriculture, Forestry, Fishing and Hunting'],
    [0, 0, 'Transportation and Warehousing'],
    [0, 0, 'Retail Trade'],
    [0, 0, 'Mining, Quarrying, and Oil and Gas Extraction'],
    [0, 0, 'Administrative and Support and Waste Management and Remediation Services'],
    [0, 0, 'Professional, Scientific, and Technical Services'],
    [0, 0, 'Information'],
    [0, 0, 'Construction'],
    [0, 0, 'Arts, Entertainment, and Recreation'],
    [0, 0, 'Health Care and Social Assistance'],
    [0, 0, 'Real Estate and Rental and Leasing'],
    [0, 0, 'Wholesale Trade'],
    [0, 0, 'Management of Companies and Enterprises'],
    [0, 0, 'Utilities'],
    [0, 0, 'Accommodation and Food Services'],
    [0, 0, 'Other Services (except Public Administration)'],
    [0, 0, 'Public Administration']
]


def append_to_csv(value, sector):
    filename = f"{sector}.csv"
    with open(filename, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([value])


symbols_url = f'https://cloud.iexapis.com/stable/ref-data/symbols?token={token}'
symbols_response = requests.get(symbols_url)
symbols_data = symbols_response.json()


for symbol_data in symbols_data:
    symbol = symbol_data['symbol']
    url = f"https://api.iex.cloud/v1/data/core/company/{symbol}?token={token}"
    url2 = f"https://api.iex.cloud/v1/data/core/advanced_stats/{symbol}?token={token}"
    data2 = requests.get(url2).json()
    data = requests.get(url).json()

    for item in data:
        for key, value in item.items():
            if key == 'sector':
                for sector_data in industries:
                    if value == sector_data[2]:
                        cash = totalCash(symbol)
                        if cash is not None:
                            sector_data[1] += cash
                            sector_data[0] += 1

                        append_to_csv(symbol, value)
                        print(industries)


csv_filename = 'industry-cash.csv'

with open(csv_filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in industries:
        csv_writer.writerow(row)
