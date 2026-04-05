#!/usr/bin/env python3
"""
Advanced filters for momentum scanner.

Apply constraints for volume, price, volatility, market cap, etc.
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumFilters:
    """Apply filters to scanner results."""

    def __init__(self):
        """Initialize filters."""
        self.applied_filters = []

    def filter_by_volume(self, results: Dict, min_volume: int = 100000) -> Dict:
        """
        Filter out low-volume stocks.

        Args:
            results: Momentum scores dict
            min_volume: Minimum volume threshold

        Returns:
            Filtered results
        """
        filtered = {
            symbol: metrics
            for symbol, metrics in results.items()
            if metrics.get('volume', 0) >= min_volume
        }

        removed = len(results) - len(filtered)
        logger.info(f"Volume filter: Removed {removed} symbols (min volume: {min_volume:,})")
        self.applied_filters.append(f'volume >= {min_volume:,}')

        return filtered

    def filter_by_price_range(self, results: Dict,
                             min_price: float = 5.0,
                             max_price: float = 1000.0) -> Dict:
        """
        Filter by price range (avoid penny stocks, expensive stocks).

        Args:
            results: Momentum scores dict
            min_price: Minimum price
            max_price: Maximum price

        Returns:
            Filtered results
        """
        filtered = {
            symbol: metrics
            for symbol, metrics in results.items()
            if min_price <= metrics.get('price', 0) <= max_price
        }

        removed = len(results) - len(filtered)
        logger.info(f"Price filter: Removed {removed} symbols "
                   f"(${min_price}-${max_price})")
        self.applied_filters.append(f'price ${min_price}-${max_price}')

        return filtered

    def filter_by_momentum_magnitude(self, results: Dict,
                                    min_magnitude: float = 0.005) -> Dict:
        """
        Filter out small momentum moves (noise).

        Args:
            results: Momentum scores dict
            min_magnitude: Minimum absolute momentum (0.005 = 0.5%)

        Returns:
            Filtered results
        """
        filtered = {
            symbol: metrics
            for symbol, metrics in results.items()
            if abs(metrics.get('intra_momentum', 0)) >= min_magnitude
        }

        removed = len(results) - len(filtered)
        logger.info(f"Magnitude filter: Removed {removed} symbols "
                   f"(min momentum: {min_magnitude*100:.2f}%)")
        self.applied_filters.append(f'momentum >= {min_magnitude*100:.2f}%')

        return filtered

    def filter_by_direction(self, results: Dict, direction: str = 'up') -> Dict:
        """
        Filter by direction (up/down/both).

        Args:
            results: Momentum scores dict
            direction: 'up', 'down', or 'both'

        Returns:
            Filtered results
        """
        if direction == 'up':
            filtered = {
                symbol: metrics
                for symbol, metrics in results.items()
                if metrics.get('intra_momentum', 0) > 0
            }
            label = "positive momentum"
        elif direction == 'down':
            filtered = {
                symbol: metrics
                for symbol, metrics in results.items()
                if metrics.get('intra_momentum', 0) < 0
            }
            label = "negative momentum"
        else:
            filtered = results
            label = "any direction"

        removed = len(results) - len(filtered)
        logger.info(f"Direction filter: Removed {removed} symbols ({label})")
        self.applied_filters.append(f'direction: {direction}')

        return filtered

    def filter_by_liquidity_score(self, results: Dict,
                                 min_score: float = 100000.0) -> Dict:
        """
        Filter by liquidity score (price × volume).

        Args:
            results: Momentum scores dict
            min_score: Minimum price × volume

        Returns:
            Filtered results
        """
        filtered = {
            symbol: metrics
            for symbol, metrics in results.items()
            if (metrics.get('price', 0) * metrics.get('volume', 0)) >= min_score
        }

        removed = len(results) - len(filtered)
        logger.info(f"Liquidity filter: Removed {removed} symbols "
                   f"(min score: {min_score:,.0f})")
        self.applied_filters.append(f'liquidity >= {min_score:,.0f}')

        return filtered

    def filter_by_blacklist(self, results: Dict,
                          blacklist: List[str] = None) -> Dict:
        """
        Filter out blacklisted symbols.

        Args:
            results: Momentum scores dict
            blacklist: List of symbols to exclude

        Returns:
            Filtered results
        """
        if not blacklist:
            blacklist = []

        filtered = {
            symbol: metrics
            for symbol, metrics in results.items()
            if symbol not in blacklist
        }

        removed = len(results) - len(filtered)
        logger.info(f"Blacklist filter: Removed {removed} symbols")
        if removed > 0:
            self.applied_filters.append(f'blacklist: {removed} excluded')

        return filtered

    def apply_all_filters(self, results: Dict,
                         config: Dict = None) -> Dict:
        """
        Apply all filters based on config.

        Args:
            results: Momentum scores dict
            config: Filter configuration dict

        Returns:
            Filtered results
        """
        if not config:
            config = {
                'min_volume': 100000,
                'min_price': 5.0,
                'max_price': 1000.0,
                'min_magnitude': 0.005,
                'direction': 'both',
                'min_liquidity_score': 100000.0,
                'blacklist': []
            }

        logger.info("Applying filters...")
        logger.info(f"Input: {len(results)} symbols")

        # Apply filters in order
        filtered = results

        if config.get('min_volume'):
            filtered = self.filter_by_volume(filtered, config['min_volume'])

        if config.get('min_price') is not None:
            filtered = self.filter_by_price_range(
                filtered,
                config.get('min_price', 5.0),
                config.get('max_price', 1000.0)
            )

        if config.get('min_magnitude'):
            filtered = self.filter_by_magnitude_threshold(
                filtered,
                config['min_magnitude']
            )

        if config.get('direction') != 'both':
            filtered = self.filter_by_direction(filtered, config['direction'])

        if config.get('min_liquidity_score'):
            filtered = self.filter_by_liquidity_score(
                filtered,
                config['min_liquidity_score']
            )

        if config.get('blacklist'):
            filtered = self.filter_by_blacklist(filtered, config['blacklist'])

        logger.info(f"Output: {len(filtered)} symbols")
        logger.info(f"Filters applied: {', '.join(self.applied_filters)}")

        return filtered

    def filter_by_magnitude_threshold(self, results: Dict,
                                     min_magnitude: float) -> Dict:
        """Helper method (same as filter_by_momentum_magnitude)."""
        return self.filter_by_momentum_magnitude(results, min_magnitude)


class RankingEngine:
    """Rank results by various metrics."""

    @staticmethod
    def rank_by_metric(results: Dict, metric: str = 'intra_momentum',
                      direction: str = 'desc') -> List[Tuple]:
        """
        Rank by any metric.

        Args:
            results: Dict of momentum results
            metric: Metric to rank by
            direction: 'desc' or 'asc'

        Returns:
            Sorted list of (symbol, metrics) tuples
        """
        reverse = direction == 'desc'
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=reverse
        )
        return ranked

    @staticmethod
    def rank_by_volume_weighted_momentum(results: Dict,
                                        volume_weight: float = 0.3) -> List[Tuple]:
        """
        Rank by momentum weighted by volume.

        Higher volume = more confidence in momentum signal.

        Args:
            results: Dict of momentum results
            volume_weight: Weight for volume component (0-1)

        Returns:
            Sorted list by weighted score
        """
        # Normalize metrics
        volumes = [m['volume'] for m in results.values()]
        momentums = [m['intra_momentum'] for m in results.values()]

        if not volumes or not momentums:
            return []

        min_vol, max_vol = min(volumes), max(volumes)
        min_mom, max_mom = min(momentums), max(momentums)

        scores = {}
        for symbol, metrics in results.items():
            vol_norm = (metrics['volume'] - min_vol) / (max_vol - min_vol + 1)
            mom_norm = (metrics['intra_momentum'] - min_mom) / (max_mom - min_mom + 1)

            weighted_score = (
                (1 - volume_weight) * mom_norm +
                volume_weight * vol_norm
            )
            scores[symbol] = weighted_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(symbol, results[symbol]) for symbol, _ in ranked]

    @staticmethod
    def rank_by_surprise_factor(results: Dict) -> List[Tuple]:
        """
        Rank by 'surprise' - big moves relative to volume/price.

        Big momentum move on high volume = high surprise.

        Args:
            results: Dict of momentum results

        Returns:
            Sorted list by surprise factor
        """
        surprise_scores = {}

        for symbol, metrics in results.items():
            # High momentum + high volume = surprise
            momentum = abs(metrics['intra_momentum'])
            volume = metrics['volume']
            price = metrics['price']

            if price > 0 and volume > 0:
                # Normalize
                surprise = momentum * (volume / price)
                surprise_scores[symbol] = surprise
            else:
                surprise_scores[symbol] = 0

        ranked = sorted(surprise_scores.items(), key=lambda x: x[1], reverse=True)
        return [(symbol, results[symbol]) for symbol, _ in ranked]
