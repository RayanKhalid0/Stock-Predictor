import requests

def get_total_cash(symbol, api_key):

    """
    Retrieve the total cash of a company based on its symbol using the Alpha Vantage API.

    Parameters:
    symbol (str): The stock symbol of the company.
    api_key (str): Your Alpha Vantage API key.

    Returns:
    float: The total cash of the company.
    """
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}"

    try:
        response = requests.get(url)
        data = response.json()
        print(data)

        total_cash = float(data['annualReports'][1]['cashAndCashEquivalentsAtCarryingValue'])
        date =(data['annualReports'][1]['fiscalDateEnding'])

        print(total_cash)

        return total_cash, date

    except Exception as e:
        print("Error:", e)
        return None



def totalCash(symbol):
    global token
    url = f"https://api.iex.cloud/v1/data/core/advanced_stats/{symbol}?token={token}"
    response2 = (requests.get(url)).json()
    print(response2)
    for item in response2:
        for key, value in item.items():
            if key == 'totalCash':
                if value is not None:
                    return value
                else:
                    return 0



# Example usage:
token = "pk_4a9546b33fc6433db703cf2ca47b7dff"
api_key = "YY2E0FPDVCIKFF5C"
symbol = "AAPL"
total_cash, date = get_total_cash(symbol, api_key)
cash = totalCash(symbol)
if total_cash is not None:
    print(f"Total cash of {symbol}: ${total_cash} for date: {date}")
    print(f"Total cash of {symbol}: ${cash}")

else:
    print("Failed to retrieve total cash.")
