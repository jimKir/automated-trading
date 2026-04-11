"""
Shared technical indicators.
Consolidates implementations used across strategy and regime modules.
"""

from __future__ import annotations

import numpy as np


def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """
    Average Directional Index (simplified Wilder smoothing).

    Returns a single ADX value for the given price arrays.
    Returns 25.0 (neutral/trending) when there is insufficient history.

    Parameters
    ----------
    highs, lows, closes : np.ndarray
        Price arrays of equal length.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    float
        ADX value. > 20 generally indicates a trending market.
    """
    n = len(highs)
    if n < period + 2:
        return 25.0

    tr_list, dm_p, dm_m = [], [], []
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hpc, lpc))

        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_p.append(up if (up > down and up > 0) else 0.0)
        dm_m.append(down if (down > up and down > 0) else 0.0)

    atr = float(np.mean(tr_list[-period:]))
    if atr == 0:
        return 0.0
    di_p = float(np.mean(dm_p[-period:])) / atr
    di_m = float(np.mean(dm_m[-period:])) / atr
    denom = di_p + di_m
    return 0.0 if denom == 0 else abs(di_p - di_m) / denom * 100
