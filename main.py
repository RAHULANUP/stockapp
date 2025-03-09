import streamlit as st
from streamlit_option_menu import option_menu
import requests
from bs4 import BeautifulSoup
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from yahooquery import search
# from yahooquery import Ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Configure page
st.set_page_config(page_title="Stocks", page_icon="💰", layout="centered")
columns_displayed = ['Scheme Name','1M','3M','6M','1Y','2Y','3Y','5Y','10Y','Weighted Return']
###Fetches mutual fund data from the given URL.
@st.cache_resource
def load_and_process():
    url = "https://www.moneycontrol.com/mutual-funds/best-funds/equity/returns/"
    response = requests.get(url)
    response.encoding = 'utf-8'
    content = response.content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(content, 'html.parser')

    # Locate the mutual funds table
    table = soup.find('table', {'id': 'dataTableId'})
    if table is None:
        raise ValueError("Mutual funds table not found on the page.")

    # Extract headers and column indices
    headers = table.find_all('th')
    columns = {}
    for idx, header in enumerate(headers):
        column_name = header.get_text(strip=True).split(' ')[0]
        columns[column_name] = idx


    required_columns = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Column '{col}' is missing from the table.")

    # Extract rows
    rows = table.find_all('tr')[1:]
    mutual_funds = []

    # Helper function to safely convert strings to floats
    def safe_float(value):
        value = value.strip().replace('%', '')
        try:
            return float(value)
        except ValueError:
            return 0.0

    # Extract data for each row
    for row in rows:
        cols = row.find_all('td')
        fund_data = {
            'Scheme Name': cols[0].text.strip(),
            '1M': safe_float(cols[columns['1M']].text),
            '3M': safe_float(cols[columns['3M']].text),
            '6M': safe_float(cols[columns['6M']].text),
            '1Y': safe_float(cols[columns['1Y']].text),
            '2Y': safe_float(cols[columns['2Y']].text),
            '3Y': safe_float(cols[columns['3Y']].text),
            '5Y': safe_float(cols[columns['5Y']].text),
            '10Y': safe_float(cols[columns['10Y']].text),
            'Link': cols[0].find('a')['href']
        }
        mutual_funds.append(fund_data)
    print(mutual_funds)
    # """
    # Convert into Dataframe
    # Adds a Weighted Return column to the DataFrame.
    # """
    df = pd.DataFrame(mutual_funds)
    weights = {'1M': 0.003, '3M': 0.005, '6M': 0.008, '1Y': 0.01, '2Y': 0.02, '3Y': 0.03, '5Y': 0.05, '10Y': 0.1}
    df['Weighted Return'] = sum(df[col] * weight for col, weight in weights.items())


    # Function to safely convert text to float
    def safe_float(value):
        value = value.strip().replace('%', '')
        try:
            return float(value)
        except ValueError:
            return 0.0
    def fetch_portfolio(fund_url):
        """
        Fetches the top 10 stock holdings from a mutual fund's portfolio with market cap classification.
        """
        response = requests.get(fund_url)
        response.encoding = 'utf-8'
        content = response.content.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(content, 'html.parser')

        table = soup.find('table', {'id': 'portfolioEquityTable'})
        if table is None:
            return []

        rows = table.find_all('tr')[1:]
        stocks = []

        for row in rows[:10]:  # Fetch only the top 10 stocks
            cols = row.find_all('td')
            stock_name = cols[0].text.strip()
            stock_link = cols[0].find('a')['href']
            sector_name = cols[1].text.strip() if len(cols) > 1 else 'Unknown'

            # Fetch stock details page to extract ticker symbol
            stock_response = requests.get(stock_link)
            stock_response.encoding = 'utf-8'
            stock_content = stock_response.content.decode('utf-8', errors='ignore')
            stock_soup = BeautifulSoup(stock_content, 'html.parser')

            # search_vals = search(stock_name)
            # quotes = search_vals.get('quotes', [])
            # ticker_symbol = quotes[0].get('symbol') if quotes else None

            # Get market cap classification
            market_cap_category = 'Unknown'
            # if ticker_symbol:
            #     try:
            #         stock = Ticker(ticker_symbol)
            #         # Get market cap from summary detail
            #         market_cap = stock.summary_detail[ticker_symbol].get('marketCap', 0)

            #         # Classification thresholds (in USD)
            #         if market_cap >= 10e9:  # $10 billion
            #             market_cap_category = 'Large Cap'
            #         elif market_cap >= 2e9:   # $2 billion
            #             market_cap_category = 'Mid Cap'
            #         else:
            #             market_cap_category = 'Small Cap'
            #     except Exception as e:print(f"Error fetching market cap for {ticker_symbol}: {str(e)}")

            stock_data = {
                'Stock Name': stock_name,
                'Sector': sector_name,
                '% of Total Holdings': safe_float(cols[3].text),
                '1M Change': safe_float(cols[4].text),
                #'Ticker Symbol': ticker_symbol,
                #'Market Cap': market_cap_category,  # New field
                'Link': stock_link
            }
            stocks.append(stock_data)

        return stocks


    # Assuming df contains mutual fund data with 'Weighted Return' and 'Link' columns
    top_funds = df.nlargest(10, 'Weighted Return')

    top_stocks = {}
    sector_stocks = {}

    for _, row in top_funds.iterrows():
        stocks = fetch_portfolio(row['Link'])
        for stock in stocks:
            weight = (row['Weighted Return'] * stock['% of Total Holdings'] * (1 + stock['1M Change'] / 100))

            if stock['Stock Name'] in top_stocks:
                top_stocks[stock['Stock Name']]['Weighted Score'] += weight
            else:
                top_stocks[stock['Stock Name']] = {
                    'Weighted Score': weight,
                    #'Ticker Symbol': stock['Ticker Symbol'],
                    #'Market Cap': stock['Market Cap']  # Store market cap
                }

            if stock['Sector'] in sector_stocks:
                sector_stocks[stock['Sector']].append({
                    'Stock Name': stock['Stock Name'],
                    #'Ticker Symbol': stock['Ticker Symbol'],
                    'Weighted Score': weight,
                    #'Market Cap': stock['Market Cap']  # Store market cap
                })
            else:
                sector_stocks[stock['Sector']] = [{
                    'Stock Name': stock['Stock Name'],
                    #'Ticker Symbol': stock['Ticker Symbol'],
                    'Weighted Score': weight,
                    #'Market Cap': stock['Market Cap']
                }]

    # Sort top stocks
    sorted_top_stocks = sorted(top_stocks.items(), key=lambda x: x[1]['Weighted Score'], reverse=True)

    # Display top 10 stocks
    print("\nTHE TOP 10 STOCKS ARE:")
    for idx, (name, data) in enumerate(sorted_top_stocks[:10], start=1):
        print(f"{idx}. {name} , Score: {data['Weighted Score']:.2f}.")

    # Display sector-wise top stocks
    print("\nTHE SECTOR-WISE TOP STOCKS:")
    for category, stocks in sector_stocks.items():
        print(f"\nCategory: {category}")
        for idx, stock in enumerate(sorted(stocks, key=lambda x: x['Weighted Score'], reverse=True)[:10], start=1):
            print(f"{idx}. Stock: {stock['Stock Name']} , Score: {stock['Weighted Score']:.2f}")

    # Create a DataFrame from the aggregated stock data
    stock_data = []

    for stock_name, stock_info in top_stocks.items():
        stock_data.append({
            'Stock Name': stock_name,
            #'Ticker Symbol': stock_info['Ticker Symbol'],
            'Weighted Score': stock_info['Weighted Score'],
            #'Market Cap': stock_info['Market Cap']
        })

    stocks_df = pd.DataFrame(stock_data)

    # Sort by Weighted Score in descending order
    stocks_df = stocks_df.sort_values(by='Weighted Score', ascending=False).reset_index(drop=True)

    # Display the DataFrame
    print("\nStock Recommendations DataFrame:")
    print(stocks_df.head(60))  
    return stocks_df,df

selected = option_menu(
    menu_title=None,
    options=["Analysis", "Prediction"],
    icons=["gear", "graph-up"],
    menu_icon="cast",
    default_index = 0,
    orientation = "horizontal"
)
stocks_df,df = load_and_process()
# 3. Main content area
if selected == "Analysis":
    # 2. Create sidebar menu
    st.write("Analysis")
    st.dataframe(df[columns_displayed])
    st.dataframe(stocks_df)

    plt.figure(figsize=(45, 22))
    sns.barplot(data=stocks_df, x='Stock Name', y='Weighted Score')
    plt.xticks(rotation=45)
    plt.title('Top Stocks by Weighted Score')
    plt.xlabel('Stock Name')
    plt.ylabel('Weighted Score')
    st.pyplot(plt)

elif selected == "Prediction":
    # 2. Create sidebar menu
    with st.sidebar:
        ticker_s =st.text_input("Enter the ticker symbol",value="COFORGE.NS")
        
    # Fetch historical data for Reliance Industries Limited (RELIANCE.NS)
    stock_symbol = "COFORGE.NS"
    data = yf.download(stock_symbol, start="2015-01-01", end="2025-01-01")

    # Prepare dataset
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    data['Daily Return'] = data['Close'].pct_change()

    # Categorize stock performance
    conditions = [
        (data['Daily Return'] > 0.01),
        (data['Daily Return'] < -0.01),
        (data['Daily Return'].between(-0.01, 0.01))
    ]
    labels = ['Good', 'Bad', 'Average']
    data['Label'] = np.select(conditions, labels)

    # Drop NaN values
    data.dropna(inplace=True)

    # Feature selection
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = data[features]
    y = data['Label']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Convert predictions back to labels
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Display results
    plt.figure(figsize=(8, 4))
    plt.hist(y_pred_labels, bins=3, edgecolor='black', alpha=0.7)
    plt.xlabel('Stock Performance')
    plt.ylabel('Frequency')
    plt.title(f'{stock_symbol} Stock Classification Results')
    plt.show()

    

