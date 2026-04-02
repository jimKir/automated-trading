#!/usr/bin/env python3
"""
Symbol lists for the Momentum Scanner V2.

Primary source: DynamicCandidateBuilder from the main trading system.
Fallback: hardcoded representative S&P 500 / Nasdaq-100 lists.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback hardcoded lists — used when DynamicCandidateBuilder is unavailable
# ---------------------------------------------------------------------------

_FALLBACK_SP500: List[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "AMD", "INTC",
    "QCOM", "CRM", "ADBE", "TXN", "AMAT", "MU", "LRCX", "PANW", "INTU",
    "NOW", "ORCL", "IBM", "HPQ", "CSCO", "SNPS", "CDNS", "KLAC", "MRVL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "USB",
    "PNC", "COF", "TFC", "MCO", "SPGI", "ICE", "CME", "CB", "MMC", "AON",
    # Healthcare
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "ISRG",
    "PFE", "GILD", "REGN", "BMY", "MDT", "CVS", "VRTX", "ELV", "HUM",
    "CI", "SYK", "BSX", "ZBH", "BAX", "BDX", "IQV",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "BKNG",
    "ABNB", "EBAY", "ORLY", "AZO", "ULTA", "CMG", "YUM", "DRI",
    # Consumer Staples
    "WMT", "PG", "COST", "PEP", "KO", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "SJM", "CPB", "HRL", "CAG",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OKE", "VLO",
    "DVN", "PXD", "HAL", "BKR", "FANG", "HES",
    # Industrials
    "CAT", "HON", "BA", "GE", "LMT", "RTX", "MMM", "ROK", "UPS", "FDX",
    "EMR", "ETN", "PH", "GD", "NOC", "LHX", "TDG", "CARR", "OTIS",
    # Materials
    "LIN", "APD", "ECL", "SHW", "PPG", "NEM", "FCX", "NUE", "VMC", "MLM",
    # Real Estate
    "PLD", "AMT", "CCI", "DLR", "EQIX", "PSA", "SPG", "O", "WELL",
    # Utilities
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "PCG", "EXC", "ED",
    # Communication Services
    "NFLX", "DIS", "T", "VZ", "CHTR", "CMCSA", "TMUS", "ATVI", "EA",
]

_FALLBACK_NASDAQ100: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "AVGO", "COST", "NFLX", "ASML", "AMD", "PEP", "LIN", "CSCO", "ADBE",
    "QCOM", "TXN", "INTU", "CMCSA", "AMAT", "ISRG", "BKNG", "MU", "HON",
    "VRTX", "LRCX", "PANW", "KLAC", "MELI", "REGN", "MDLZ", "CDNS",
    "SNPS", "ABNB", "ORLY", "FTNT", "CRWD", "ROP", "CTAS", "MNST", "MRVL",
    "KDP", "PCAR", "ADP", "PAYX", "WDAY", "DXCM", "ODFL", "FAST", "BIIB",
    "DLTR", "IDXX", "VRSK", "ANSS", "ALGN", "TEAM", "ZS", "ILMN", "CPRT",
    "ROST", "GILD", "PLTR", "EBAY", "PYPL", "INTC", "PDD", "CEG", "GFS",
    "FANG", "ON", "TTWO", "DDOG", "SNOW", "COIN", "RBLX", "TTD", "APP",
    "MSTR", "SMCI", "DECK", "AXON", "NTRA", "PODD", "GEHC", "CDW", "CCEP",
    "PSTG", "TXRH",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sp500_symbols() -> List[str]:
    """
    Get S&P 500 + Nasdaq-100 symbols via DynamicCandidateBuilder.
    Falls back to hardcoded list if main system not available.
    """
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from strategy.universe import DynamicCandidateBuilder
        # Minimal config for symbol fetch only
        cfg = {
            "dynamic_candidates": {
                "enabled": True,
                "min_avg_volume_usd": 5_000_000,
                "min_history_days": 252,
            }
        }
        builder = DynamicCandidateBuilder(cfg)
        syms = builder._fetch_constituents(verbose=False)
        if syms and len(syms) > 50:
            logger.info(f"DynamicCandidateBuilder: {len(syms)} symbols")
            return syms
    except Exception as e:
        logger.debug(f"DynamicCandidateBuilder unavailable: {e}")
    # Fallback to hardcoded representative list
    return _FALLBACK_SP500


def get_nasdaq100_symbols() -> List[str]:
    """
    Get Nasdaq-100 symbols via DynamicCandidateBuilder.
    Falls back to hardcoded list if main system not available.
    """
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from strategy.universe import DynamicCandidateBuilder
        # Minimal config — Nasdaq-100 only
        cfg = {
            "dynamic_candidates": {
                "enabled": True,
                "include_sp500": False,
                "include_ndx100": True,
                "min_avg_volume_usd": 5_000_000,
                "min_history_days": 252,
            }
        }
        builder = DynamicCandidateBuilder(cfg)
        syms = builder._fetch_constituents(verbose=False)
        if syms and len(syms) > 50:
            logger.info(f"DynamicCandidateBuilder (NDX100): {len(syms)} symbols")
            return syms
    except Exception as e:
        logger.debug(f"DynamicCandidateBuilder unavailable: {e}")
    # Fallback to hardcoded Nasdaq-100 list
    return _FALLBACK_NASDAQ100


def get_combined_symbols() -> List[str]:
    """
    Return de-duplicated union of S&P 500 and Nasdaq-100 symbols.
    """
    sp500  = get_sp500_symbols()
    ndx100 = get_nasdaq100_symbols()
    seen: set = set(sp500)
    combined = list(sp500)
    for sym in ndx100:
        if sym not in seen:
            seen.add(sym)
            combined.append(sym)
    return combined
