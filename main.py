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
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime,timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# 1. Configure pages
st.set_page_config(page_title="WiseFunds", page_icon="🤝", layout="centered")
columns_displayed = ['Scheme Name','1M','3M','6M','1Y','2Y','3Y','5Y','10Y','Weighted Return']


@st.cache_resource(ttl=3600)
def load_and_process():
    url = "https://www.moneycontrol.com/mutual-funds/best-funds/equity/returns/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'dataTableId'})
    headers = table.find_all('th')
    columns = {header.get_text(strip=True).split(' ')[0]: idx for idx, header in enumerate(headers)}

    required_columns = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Column '{col}' missing.")
    
    # Helper function to safely convert string to float
    def safe_float(value):
        value = value.strip().replace('%', '')
        if value == '-' or value == '' or value == 'NA':
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    rows = table.find_all('tr')[1:]
    mutual_funds = []
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

    df = pd.DataFrame(mutual_funds)
    weights = {'1M': 0.003, '3M': 0.005, '6M': 0.008, '1Y': 0.01, '2Y': 0.02, '3Y': 0.03, '5Y': 0.05, '10Y': 0.1}
    df['Weighted Return'] = sum(df[col] * weight for col, weight in weights.items())
    
    @st.cache_data(ttl=3600)
    def fetch_portfolio(fund_url):
        response = requests.get(fund_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'portfolioEquityTable'})
        if not table:
                return []
                
        rows = table.find_all('tr')[1:11]  # Only get top 10 stocks 
        
        stocks = []
        for row in rows[:5]:  # Process top 5 stocks
            cols = row.find_all('td')
            stocks.append({
                'Stock Name': cols[0].text.strip(),
                'Sector': cols[1].text.strip() if len(cols) > 1 else 'Unknown',
                '% of Total Holdings': safe_float(cols[3].text),
                '1M Change': safe_float(cols[4].text),
            })
        return stocks

    # Process top 5 funds instead of 10
    top_funds = df.nlargest(5, 'Weighted Return')
    top_stocks = {}
    sector_stocks = {}

    for _, row in top_funds.iterrows():
        stocks = fetch_portfolio(row['Link'])
        for stock in stocks:
            weight = row['Weighted Return'] * stock['% of Total Holdings'] * (1 + stock['1M Change'] / 100)
            stock_name = stock['Stock Name']
            
            # Update top_stocks
            if stock_name in top_stocks:
                top_stocks[stock_name]['Weighted Score'] += weight
            else:
                top_stocks[stock_name] = {'Weighted Score': weight}
            
            # Update sector_stocks
            sector = stock['Sector']
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append({'Stock Name': stock_name, 'Weighted Score': weight})

    # Create dataframes
    stocks_df = pd.DataFrame([
        {'Stock Name': name, 'Weighted Score': data['Weighted Score']}
        for name, data in top_stocks.items()
    ]).sort_values('Weighted Score', ascending=False)

    sector_df = pd.DataFrame([
        {'Sector': sector, 'Stock Name': stock['Stock Name'], 'Weighted Score': stock['Weighted Score']}
        for sector, stocks in sector_stocks.items()
        for stock in sorted(stocks, key=lambda x: x['Weighted Score'], reverse=True)[:5]  # Top 5 per sector
    ])

    return stocks_df, df, sector_df

selected = option_menu(
    menu_title=None,
    options=["Analysis", "Prediction"],
    icons=["robot", "graph-up"],
    menu_icon="cast",
    default_index = 0,
    orientation = "horizontal"
)



stocks_df,df,sector_stocks = load_and_process()


def improve_top_stocks_visualization(stocks_df):
    top_10_stocks = stocks_df.head(10)

    fig = px.bar(
        top_10_stocks,
        x='Stock Name',
        y='Weighted Score',
        title='Top 10 Stocks by Weighted Score',
        labels={'Weighted Score': 'Performance Score', 'Stock Name': 'Stock'},
        color='Weighted Score',
        color_continuous_scale='blues',
    )

    fig.update_traces(
        marker=dict(line=dict(width=2, color='black')),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}',
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        width=900,
        title_x=0.5,
        font=dict(size=14, family="Arial"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)

# Improve Sector Distribution Visualization
def improve_sector_distribution(sector_stocks):
    sector_counts = sector_stocks.groupby('Sector').size().reset_index(name='Count')

    fig_count = px.pie(
        sector_counts,
        values='Count',
        names='Sector',
        title='Stock Distribution by Sector',
        hole=0.35,
        color_discrete_sequence=px.colors.sequential.Bluered,
    )

    fig_count.update_traces(
        textinfo='percent+label',
        pull=[0.1 if i == sector_counts['Count'].idxmax() else 0 for i in range(len(sector_counts))],
    )

    sector_avg = sector_stocks.groupby('Sector')['Weighted Score'].mean().reset_index()
    sector_avg = sector_avg.sort_values('Weighted Score', ascending=False)

    fig_performance = px.bar(
        sector_avg,
        x='Sector',
        y='Weighted Score',
        title='Average Performance by Sector',
        labels={'Weighted Score': 'Avg. Performance Score'},
        color='Weighted Score',
        color_continuous_scale='blues',
    )

    fig_performance.update_layout(
        xaxis_tickangle=-45,
        height=600,
        width=900,
        title_x=0.5,
        font=dict(size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_count, use_container_width=True)
    st.plotly_chart(fig_performance, use_container_width=True)

# Improve Fund Performance Visualization
def improve_fund_performance(df, time_periods):
    top_funds = df.nlargest(5, 'Weighted Return')

    fig = go.Figure()

    for period in time_periods:
        fig.add_trace(go.Scatter(
            x=top_funds['Scheme Name'],
            y=top_funds[period],
            mode='lines+markers',
            name=f'{period} Returns',
            line=dict(width=3),
            marker=dict(size=8, line=dict(width=2, color='black')),
        ))

    fig.update_layout(
        title='Top 5 Funds Performance Across Time Periods',
        xaxis_title='Fund Name',
        yaxis_title='Returns (%)',
        height=600,
        width=900,
        title_x=0.5,
        font=dict(size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)

# Main visualization function
def enhanced_visualizations(stocks_df, df, sector_stocks):
    viz_tabs = st.tabs(["Top Stocks", "Sector Distribution", "Fund Performance"])

    with viz_tabs[0]:  # Top Stocks
        improve_top_stocks_visualization(stocks_df)

    with viz_tabs[1]:  # Sector Distribution
        improve_sector_distribution(sector_stocks)

    with viz_tabs[2]:  # Fund Performance
        time_periods = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']
        selected_periods = st.multiselect(
            "Select time periods to compare",
            time_periods,
            default=['1Y', '3Y']
        )
        improve_fund_performance(df, selected_periods)

# 3. Main content area
if selected == "Analysis":
    # 2. Create sidebar menu
    with st.sidebar:
        uchoice = option_menu(
            menu_title=None,
            options=["All Funds","All_Stocks","Sector Stocks","Visualizer","Top 5 Funds","Top 5 Stocks"],
            icons=["database","bar-chart","layers","pie-chart","kanban-fill","kanban-fill"],
            menu_icon = "cast",
            default_index = 0
        )
    if uchoice == "All Funds":
        st.subheader("All Funds")
        st.dataframe(df[columns_displayed],hide_index=True)
    elif uchoice == "All_Stocks":
        st.subheader("All Stocks")
        st.dataframe(stocks_df,hide_index=True)
    elif uchoice == "Sector Stocks":
        st.dataframe(sector_stocks,hide_index=True)
    elif uchoice == "Visualizer":
        enhanced_visualizations(stocks_df,df,sector_stocks)
    elif uchoice == "Top 5 Funds":
        st.subheader("Top 5 Mutual Funds")
        top_funds = df[["Scheme Name","Weighted Return"]].nlargest(5,"Weighted Return")
        st.dataframe(top_funds,hide_index=True)
    elif uchoice == "Top 5 Stocks":
        st.subheader("Top 5 Stocks")
        top_stocks = stocks_df[["Stock Name","Weighted Score"]].head(5)
        st.dataframe(top_stocks,hide_index=True)

elif selected == "Prediction":
    with st.sidebar:
        symbol = st.text_input("Enter the ticker symbol", value="COFORGE.NS")
        choice = st.selectbox("Choose an option", ["Features", "Prediction"])

    @st.cache_resource  
    def fetch_stock_data(ticker, period='2y'):
        """
        Fetch historical data for a given ticker using yfinance.
        
        Args:
            ticker (str): Stock symbol with suffix (e.g., .NS for NSE).
            period (str): Time period to fetch data for.
        
        Returns:
            DataFrame: Historical data or None if no data is fetched.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist if not hist.empty else None
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None

    def calculate_features(df):
        """
        Calculate technical indicators as features.
        
        Returns:
            DataFrame: DataFrame with additional columns for technical indicators.
        """
        df = df.copy()

        # Return-based features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Weekly_Return'] = df['Close'].pct_change(periods=5)
        df['Monthly_Return'] = df['Close'].pct_change(periods=21)
        df['Quarterly_Return'] = df['Close'].pct_change(periods=63)
        df['Yearly_Return'] = df['Close'].pct_change(periods=252)

        # 30-day rolling volatility
        df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std()

        # Moving averages and ratio
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        df['MA_Ratio'] = df['MA_50'] / df['MA_200']

        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA_20']

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Drop NaNs and reset index
        return df.dropna().reset_index(drop=True)

    def create_labels(df, forward_period=126):
        """
        Create labels based on forward returns over a given period.
        
        Args:
            df (DataFrame): Stock historical data.
            forward_period (int): Days to calculate forward return (default ~6 months).
        
        Returns:
            DataFrame: DataFrame with a new 'Label' column.
        """
        df = df.copy()
        df['Forward_Return'] = df['Close'].shift(-forward_period) / df['Close'] - 1
        df = df.dropna(subset=['Forward_Return'])
        df['Label'] = pd.qcut(df['Forward_Return'], 3, labels=['bad', 'average', 'good'], duplicates='drop')
        return df.dropna(subset=['Label'])

    def prepare_model_data(tickers_list):
        """
        Prepare and combine model data from a list of tickers.
        
        Returns:
            DataFrame: Combined DataFrame with technical indicators, labels, and ticker info.
        """
        data_frames = []
        for ticker in tickers_list:
            df = fetch_stock_data(ticker)
            if df is None:
                continue
            df = calculate_features(df)
            df = create_labels(df)
            df['Ticker'] = ticker
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
        
    @st.cache_resource    
    def build_model(data):
        """
        Build and train the Random Forest model using hyperparameter tuning.
        
        Uses GridSearchCV with parallel processing (n_jobs=-1) and displays the best parameters.
        
        Returns:
            tuple: Trained model, the fitted scaler, and evaluation outputs.
        """
        data = data.dropna(subset=['Label'])
        features = ['Daily_Return', 'Weekly_Return', 'Monthly_Return',
                    'Quarterly_Return', 'Yearly_Return', 'Volatility_30d',
                    'MA_Ratio', 'RSI', 'Relative_Volume']
        X = data[features]
        y = data['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # Format best parameters: convert None to "Unlimited" for clarity
        best_params = grid_search.best_params_
        display_params = {k: ("Unlimited" if v is None else v) for k, v in best_params.items()}
        
        # st.markdown("*Best Hyperparameters:*")
        # output_md = ""
        # for param, val in display_params.items():
        #     output_md += f"- *{param}*: {val}\n"
        # st.markdown(output_md)

        # Evaluate the model
        y_pred = best_model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        # st.subheader("📊 Classification Report")
        # st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
        # st.subheader("🗓️ Confusion Matrix")
        # st.write(confusion_matrix(y_test, y_pred))

        # Compute and display feature importances
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        # st.subheader("🔥 Feature Importance")
        # cols = st.columns(3)
        # for i in range(len(features)):
        #     column = cols[i % 3]
        #     column.write(f"*{features[indices[i]]}*: {importances[indices[i]]:.4f}")
        
        return best_model, scaler, df_report, importances, indices, features, y_test, y_pred, display_params

    def predict_stock(model, scaler, ticker):
        """
        Predict the classification for a new stock.
        
        Returns:
            dict: Contains the ticker, predicted class, and class probabilities.
        """
        df = fetch_stock_data(ticker)
        if df is None:
            return None
        df = calculate_features(df)
        if df.empty:
            return None
        latest_data = df.iloc[-1:]
        features = ['Daily_Return', 'Weekly_Return', 'Monthly_Return',
                    'Quarterly_Return', 'Yearly_Return', 'Volatility_30d',
                    'MA_Ratio', 'RSI', 'Relative_Volume']
        X = latest_data[features]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        return {
            'ticker': ticker,
            'prediction': prediction,
            'probabilities': {
                'bad': probabilities[0],
                'average': probabilities[1],
                'good': probabilities[2]
            }
        }
        
    # List of tickers for training (example list for the Indian market)
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS"
    ]

    print("### Data Preparation Phase")
    data = prepare_model_data(tickers)
    if data.empty:
        st.error("No data available for model training.")
    else:
        print("### Model Training Phase with Parameter Tuning")
        model, scaler, df_report, importances, indices, features, y_test, y_predict, display_params = build_model(data)

        if choice == "Features":
            print("Model Evaluation Completed. Review the classification report, confusion matrix, and feature importances above.")
            st.header("📊 Classification Report")
            st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
            st.header("🗓️ Confusion Matrix")
            st.write(confusion_matrix(y_test, y_predict))

            st.header("🔥 Feature Importance")
            cols = st.columns(3)
            for i in range(len(features)):
                column = cols[i % 3]
                column.write(f"*{features[indices[i]]}*: {importances[indices[i]]:.4f}")

            st.header("Best Hyperparameters:")
            with st.container():
                for param, val in display_params.items():
                    st.markdown(f"🔹 **{param}**: `{val}`")

        elif choice == "Prediction":
            result = predict_stock(model, scaler, symbol)
            if result is None:
                st.error("Unable to generate prediction for the entered ticker.")
            else:
                st.subheader(f"Prediction for {symbol}:")
                if result['prediction'] == 'good':
                    st.success("Classification: GOOD STOCK!")
                elif result['prediction'] == 'bad':
                    st.error("Classification: BAD STOCK!")
                else:
                    st.warning("Classification: AVERAGE STOCK!")
                categories = ['Bad', 'Average', 'Good']
                values = [
                    result['probabilities']['bad'],
                    result['probabilities']['average'],
                    result['probabilities']['good']
                ]
                # Create bar chart using Plotly Express
                fig = px.bar(x=categories, y=values, labels={'x': 'Classification', 'y': 'Probability'},
                            title="Prediction Probabilities", color=categories,
                            color_discrete_map={'Bad': '#FF5252', 'Average': '#FFC107', 'Good': '#4CAF50'})
                st.plotly_chart(fig)

                
            # st.subheader(f"Bad: {result['probabilities']['bad']:.2f}")
            # st.subheader(f"Average: {result['probabilities']['average']:.2f}")
            # st.subheader(f"Good: {result['probabilities']['good']:.2f}"
            
            

