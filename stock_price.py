import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbol for AAPL
ticker_symbol = "MSFT"

# Get the data from Yahoo Finance
data = yf.download(ticker_symbol, period="1y")

# Extract the closing prices
closing_prices = data["Close"]

# Plot the closing prices
plt.plot(closing_prices)
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("MSFT Price Chart")
plt.grid(True)

# Display the chart within VS Code
plt.show()