#!/usr/bin/env python3
"""
Symbol universe definitions.

Predefined universes: S&P 500, Sectors, All US Equities.
"""

import logging

logger = logging.getLogger(__name__)


def get_symbol_list(universe: str = "sp500") -> list[str]:
    """
    Get list of symbols for universe.

    Args:
        universe: 'sp500', 'sectors', 'nasdaq100', 'all'

    Returns:
        List of symbols
    """
    if universe == "sp500":
        return get_sp500_symbols()
    if universe == "sectors":
        return get_sector_leaders()
    if universe == "nasdaq100":
        return get_nasdaq100_symbols()
    if universe == "all":
        return get_all_symbols()
    if universe.endswith(".csv"):
        return load_symbols_from_csv(universe)
    logger.warning(f"Unknown universe: {universe}, defaulting to S&P 500")
    return get_sp500_symbols()


def get_sp500_symbols() -> list[str]:
    """Get S&P 500 symbols."""
    # Top 500 most liquid US stocks
    # This is a representative sample - in production, fetch from:
    # https://www.slickcharts.com/sp500 or similar

    return [
        # Top mega-cap tech
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "GOOG",
        "META",
        "AMZN",
        "TSLA",
        # Finance
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "BLK",
        "SCHW",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "LLY",
        "CVS",
        # Industrials
        "BA",
        "CAT",
        "GE",
        "MMM",
        "RTX",
        "LMT",
        "HON",
        "ROK",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "MPC",
        "PSX",
        "OKE",
        "SLB",
        "FANG",
        # Consumer
        "WMT",
        "COST",
        "HD",
        "TGT",
        "LOW",
        "MCD",
        "NKE",
        "SBUX",
        # Real Estate
        "PLD",
        "DLR",
        "EQIX",
        "WELL",
        "AVB",
        "PTC",
        "ARE",
        "PSA",
        # Utilities
        "NEE",
        "DUK",
        "SO",
        "D",
        "EXC",
        "LNT",
        "AEP",
        "XEL",
        # Communication
        "VZ",
        "T",
        "CMCSA",
        "CHTR",
        "TMUS",
        # Materials
        "LIN",
        "APD",
        "SHW",
        "NUE",
        "IP",
        "KIM",
        "FCX",
        "CF",
        # Semiconductors
        "AMD",
        "QCOM",
        "AVGO",
        "MU",
        "INTC",
        "MCHP",
        "NXPI",
        "KLAC",
        # Software
        "CRM",
        "ADBE",
        "ORCL",
        "SAP",
        "SNOW",
        "ZOOM",
        "OKTA",
        "DDOG",
        # Additional liquid names
        "PG",
        "KO",
        "PEP",
        "CSCO",
        "IBM",
        "ACN",
        "SYK",
        "ISRG",
        "AXP",
        "INTU",
        "AMAT",
        "ASML",
        "LRCX",
        "ADI",
        "TXN",
        "SWKS",
        # Add 450+ more for full S&P 500...
        # For brevity, truncating here - use real S&P 500 list in production
    ]


def get_sector_leaders() -> list[str]:
    """Get top leader from each major sector."""
    return [
        # Technology
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "LLY",
        # Financials
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        # Industrials
        "BA",
        "CAT",
        "GE",
        "MMM",
        "RTX",
        # Consumer Discretionary
        "AMZN",
        "TSLA",
        "WMT",
        "HD",
        "MCD",
        # Consumer Staples
        "PG",
        "KO",
        "PEP",
        "COST",
        "NKE",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "MPC",
        "PSX",
        # Materials
        "LIN",
        "APD",
        "SHW",
        "NUE",
        "FCX",
        # Real Estate
        "PLD",
        "DLR",
        "EQIX",
        "WELL",
        "AVB",
        # Communication Services
        "VZ",
        "T",
        "CMCSA",
        "CHTR",
        "TMUS",
        # Utilities
        "NEE",
        "DUK",
        "SO",
        "D",
        "EXC",
    ]


def get_nasdaq100_symbols() -> list[str]:
    """Get Nasdaq 100 symbols."""
    return [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "ASML",
        "AVGO",
        "AMAT",
        "AMD",
        "ADSK",
        "ADP",
        "ABNB",
        "ALTERA",
        "ANSS",
        "ASHR",
        "ADI",
        "ADBE",
        "ARM",
        "ANET",
        "BIDU",
        "BIIB",
        "BLKC",
        "BKNG",
        "CDNS",
        "CERNER",
        "CHKP",
        "CHWY",
        "CRM",
        "CSCO",
        "COIN",
        "CCXI",
        "CSGP",
        "CTSH",
        "CMCSA",
        "CSX",
        "CPRT",
        "CRWD",
        "DDOG",
        "DLTR",
        "DXC",
        "DPZ",
        "EBAY",
        "EXC",
        "EXPE",
        "EXEQ",
        "FAST",
        "FISV",
        "FTNT",
        "FTCH",
        "GILD",
        "GOOG",
        "GRMN",
        "GLAYF",
        "GRUB",
        "HCAT",
        "HSTM",
        "HSIC",
        "HUBB",
        "HWKN",
        "HYLN",
        "ILMN",
        "INTU",
        "INTC",
        "INMD",
        "IONQ",
        "JBLU",
        "JKHY",
        "JCOM",
        "KLAC",
        "KDP",
        "KEYS",
        "LRCX",
        "LOGI",
        "LILM",
        "LYV",
        "MARVL",
        "MASTR",
        "MCHP",
        "MDLZ",
        "MRNA",
        "MNST",
        "MSFT",
        "MU",
        "MXIM",
        "NFLX",
        "NXPI",
        "NVDA",
        "NEON",
        "ODFL",
        "OKTA",
        "ORCL",
        "PAYX",
        "PCAR",
        "PENN",
        "PKOH",
        "PYPL",
        "PEP",
        "PSEC",
        "PLTR",
        "QCOM",
        "QRVO",
        "RVNC",
        "RBLX",
        "ROIV",
        "ROKU",
        "ROST",
        "SAIC",
        "SGEN",
        "SIRI",
        "SMCI",
        "SPLK",
        "SNOW",
        "SNPS",
        "STWD",
        "SWKS",
        "TEAM",
        "TMDX",
        "TDOC",
        "TLIS",
        "TMUZ",
        "TTEK",
        "TROW",
        "TSLA",
        "TWTR",
        "TXNN",
        "TTD",
        "UBER",
        "ULTA",
        "VRSN",
        "VRSK",
        "VRTX",
        "WDAY",
        "WDRV",
        "WERN",
        "WKME",
        "WUSH",
        "XCEL",
        "XONC",
        "YEXT",
        "ZLAB",
        "ZOOM",
        "ZSCALER",
    ]


def get_all_symbols() -> list[str]:
    """
    Get all US equities (simplified).

    Note: In production, fetch from Russell 3000 or similar.
    This is a large subset for demonstration.
    """
    # This would ideally load from a file or API
    # For now, return a large representative sample

    all_symbols = get_sp500_symbols()  # Start with S&P 500

    # Add popular mid/small-cap stocks
    additional = [
        # Mid-caps
        "AFRM",
        "BGFV",
        "BOOT",
        "BURL",
        "CATO",
        "CBRL",
        "CELH",
        "CFLT",
        "CHPT",
        "CLPS",
        "CSGP",
        "DASH",
        "DENN",
        "DKNG",
        "DMTK",
        "DOCU",
        "DOMO",
        "DOOR",
        "DOORDASH",
        "DUKEMI",
        "DUOL",
        "EAT",
        "ELEV",
        # Add thousands more in production...
    ]

    all_symbols.extend(additional)
    return list(set(all_symbols))  # Remove duplicates


def load_symbols_from_csv(csv_file: str) -> list[str]:
    """
    Load symbols from CSV file.

    Expected format: One symbol per line, or CSV with 'symbol' column.
    """
    try:
        with open(csv_file) as f:
            lines = f.readlines()

        symbols = []
        for line in lines:
            # Handle CSV with headers
            if line.lower().startswith("symbol"):
                continue
            # Get first column
            symbol = line.split(",")[0].strip().upper()
            if symbol and len(symbol) <= 5:  # Valid stock symbol
                symbols.append(symbol)

        logger.info(f"Loaded {len(symbols)} symbols from {csv_file}")
        return symbols

    except Exception as e:
        logger.error(f"Error loading symbols from CSV: {e}")
        return []


def validate_symbols(symbols: list[str]) -> list[str]:
    """
    Validate symbol list.

    Returns only valid symbols.
    """
    valid = []
    for symbol in symbols:
        symbol = symbol.upper().strip()
        # Basic validation: 1-5 letters
        if len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalpha():
            valid.append(symbol)

    return valid
