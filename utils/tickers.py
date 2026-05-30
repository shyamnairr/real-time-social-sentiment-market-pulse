"""
Ticker universe (50+ assets) and a name -> ticker lookup map.

Used by:
  - entity_extraction.py  (to map "$AAPL" and "Apple" -> AAPL)
  - synthetic_generator.py (to know which assets to simulate)
  - data collectors / yfinance price pulls
"""

# (ticker, company name, sector) -- 55 widely-discussed US equities + a few ETFs.
TICKER_UNIVERSE = [
    ("AAPL", "Apple", "Technology"),
    ("MSFT", "Microsoft", "Technology"),
    ("GOOGL", "Alphabet", "Technology"),
    ("AMZN", "Amazon", "Consumer Discretionary"),
    ("NVDA", "Nvidia", "Technology"),
    ("META", "Meta Platforms", "Technology"),
    ("TSLA", "Tesla", "Consumer Discretionary"),
    ("AMD", "Advanced Micro Devices", "Technology"),
    ("NFLX", "Netflix", "Communication Services"),
    ("INTC", "Intel", "Technology"),
    ("JPM", "JPMorgan Chase", "Financials"),
    ("BAC", "Bank of America", "Financials"),
    ("WFC", "Wells Fargo", "Financials"),
    ("GS", "Goldman Sachs", "Financials"),
    ("V", "Visa", "Financials"),
    ("MA", "Mastercard", "Financials"),
    ("DIS", "Disney", "Communication Services"),
    ("KO", "Coca-Cola", "Consumer Staples"),
    ("PEP", "PepsiCo", "Consumer Staples"),
    ("WMT", "Walmart", "Consumer Staples"),
    ("COST", "Costco", "Consumer Staples"),
    ("MCD", "McDonald's", "Consumer Discretionary"),
    ("NKE", "Nike", "Consumer Discretionary"),
    ("SBUX", "Starbucks", "Consumer Discretionary"),
    ("BA", "Boeing", "Industrials"),
    ("CAT", "Caterpillar", "Industrials"),
    ("GE", "General Electric", "Industrials"),
    ("XOM", "Exxon Mobil", "Energy"),
    ("CVX", "Chevron", "Energy"),
    ("PFE", "Pfizer", "Health Care"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("UNH", "UnitedHealth", "Health Care"),
    ("MRNA", "Moderna", "Health Care"),
    ("ABBV", "AbbVie", "Health Care"),
    ("CRM", "Salesforce", "Technology"),
    ("ORCL", "Oracle", "Technology"),
    ("ADBE", "Adobe", "Technology"),
    ("CSCO", "Cisco", "Technology"),
    ("QCOM", "Qualcomm", "Technology"),
    ("PYPL", "PayPal", "Financials"),
    ("XYZ", "Block", "Financials"),
    ("SHOP", "Shopify", "Technology"),
    ("UBER", "Uber", "Technology"),
    ("ABNB", "Airbnb", "Consumer Discretionary"),
    ("COIN", "Coinbase", "Financials"),
    ("PLTR", "Palantir", "Technology"),
    ("SOFI", "SoFi", "Financials"),
    ("GME", "GameStop", "Consumer Discretionary"),
    ("AMC", "AMC Entertainment", "Communication Services"),
    ("F", "Ford", "Consumer Discretionary"),
    ("GM", "General Motors", "Consumer Discretionary"),
    ("T", "AT&T", "Communication Services"),
    ("VZ", "Verizon", "Communication Services"),
    ("SPY", "S&P 500 ETF", "ETF"),
    ("QQQ", "Nasdaq 100 ETF", "ETF"),
]

# Convenience lists / maps -------------------------------------------------
TICKERS = [t[0] for t in TICKER_UNIVERSE]
TICKER_TO_NAME = {t[0]: t[1] for t in TICKER_UNIVERSE}
TICKER_TO_SECTOR = {t[0]: t[2] for t in TICKER_UNIVERSE}

# Lowercased company-name -> ticker, for fuzzy entity extraction.
# (only includes distinctive names to avoid false hits like "ford" the verb)
NAME_TO_TICKER = {name.lower(): tick for tick, name, _ in TICKER_UNIVERSE}

# Common words that look like tickers but usually aren't, to reduce noise
# when a cashtag is missing (e.g. "A", "I", "BE", "ON", "GO").
AMBIGUOUS_TICKERS = {"A", "I", "BE", "ON", "GO", "OR", "SO", "IT", "ALL", "ALL"}


def is_valid_ticker(symbol: str) -> bool:
    """Return True if `symbol` is in our tracked universe."""
    return symbol.upper() in TICKER_TO_NAME
