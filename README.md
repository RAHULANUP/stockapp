# WiseFunds

WiseFunds is a financial analysis and visualization application built using Streamlit. It helps users analyze mutual fund returns, visualize stock trends, and explore sector-wise investment distributions.

## Features
- *Mutual Fund Performance Analysis:*

  - Scrapes mutual fund return data from MoneyControl.
  - Calculates weighted returns based on different time periods.

- *Stock Portfolio Insights:*

  - Extracts top 5 stocks from the best-performing mutual funds.
  - Aggregates sector-wise distributions and stock-wise scores.

- *Interactive Visualizations:*

  - Top 10 stocks by weighted score.
  - Sector distribution pie chart.
  - Fund performance comparison across time periods.

## Technologies Used

- *Frontend:* Streamlit
- - *Data Processing:* Pandas, NumPy, BeautifulSoup
- *Machine Learning:* Scikit-learn (for predictions)
- *Visualization:* Plotly, Seaborn, Matplotlib
- *Data Source:* MoneyControl (Web Scraping)

## Installation

### Prerequisites

Ensure you have Python installed. Then, install dependencies using:

sh
pip install -r requirements.txt


### Running the Application

sh
streamlit run app.py


## Usage

1. Run the Streamlit app.
2. Navigate through the tabs:
   - *Analysis:* View mutual funds, stock insights, and sector-wise distributions.
   - *Prediction:* Utilize machine learning models for financial forecasting.
3. Use interactive visualizations to explore fund performances and sector distributions.

## Customization

Modify the weighting factors in load_and_process() to adjust the weighted return calculations:

python
weights = {'1M': 0.003, '3M': 0.005, '6M': 0.008, '1Y': 0.01, '2Y': 0.02, '3Y': 0.03, '5Y': 0.05, '10Y': 0.1}


## Future Enhancements

- Add support for more financial metrics.
- Improve ML models for better predictions.
- Integrate alternative data sources for broader analysis.

## License

MIT License

## Author

Dominic Prince, Nazim Filzer, Parthiv Anil, Rahul Anup Varma
