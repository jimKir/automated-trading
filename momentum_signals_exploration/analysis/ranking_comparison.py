"""
Ranking Methods Comparison Tool

Analyzes performance of different ranking methods:
- Standard ranking (momentum score)
- Volume-weighted momentum
- Surprise factor (big moves on high volume)
- Custom hybrid rankings

Usage:
    python ranking_comparison.py --universe sp500 --hours 24
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import MomentumFilters, RankingEngine
from scanner import MomentumScanner


class RankingComparison:
    """Compare performance of different ranking methods."""

    def __init__(self, data_source: str = "alpaca"):
        self.scanner = MomentumScanner(data_source=data_source)
        self.ranking_engine = RankingEngine()
        self.filters = MomentumFilters()
        self.results = {}

    def run_scan(self, universe: str = "sp500", num_symbols: int = None) -> list[dict]:
        """Run momentum scan on universe."""
        print(f"📊 Scanning {universe}...")
        return self.scanner.run_full_scan(universe, num_symbols or 100)

    def compare_rankings(self, scan_results: list[dict]) -> dict:
        """Compare all ranking methods on same scan results."""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols_scanned": len(scan_results),
            "rankings": {},
        }

        # Method 1: Standard ranking
        print("\n🔍 Method 1: Standard Ranking (by momentum score)")
        standard = self.ranking_engine.rank_by_metric(scan_results, metric="combined")
        comparison["rankings"]["standard"] = {
            "method": "Standard momentum score",
            "top_5": [
                {
                    "symbol": r.get("symbol"),
                    "momentum": round(r.get("combined_momentum", 0), 6),
                    "volume": r.get("volume", 0),
                    "return_pct": round(r.get("hourly_return_pct", 0), 3),
                }
                for r in standard[:5]
            ],
            "bottom_5": [
                {
                    "symbol": r.get("symbol"),
                    "momentum": round(r.get("combined_momentum", 0), 6),
                    "volume": r.get("volume", 0),
                    "return_pct": round(r.get("hourly_return_pct", 0), 3),
                }
                for r in standard[-5:]
            ],
        }
        print(
            f"  Top symbol: {standard[0].get('symbol')} (momentum: {standard[0].get('combined_momentum', 0):.6f})"
        )

        # Method 2: Volume-weighted momentum
        print("\n🔍 Method 2: Volume-Weighted Momentum")
        vol_weighted = self.ranking_engine.rank_by_volume_weighted_momentum(scan_results)
        comparison["rankings"]["volume_weighted"] = {
            "method": "Volume-weighted momentum",
            "top_5": [
                {
                    "symbol": r.get("symbol"),
                    "weighted_momentum": round(
                        r.get("combined_momentum", 0) * (r.get("volume", 1) / 1000000), 6
                    ),
                    "volume": r.get("volume", 0),
                    "return_pct": round(r.get("hourly_return_pct", 0), 3),
                }
                for r in vol_weighted[:5]
            ],
            "bottom_5": [
                {
                    "symbol": r.get("symbol"),
                    "weighted_momentum": round(
                        r.get("combined_momentum", 0) * (r.get("volume", 1) / 1000000), 6
                    ),
                    "volume": r.get("volume", 0),
                    "return_pct": round(r.get("hourly_return_pct", 0), 3),
                }
                for r in vol_weighted[-5:]
            ],
        }
        print(
            f"  Top symbol: {vol_weighted[0].get('symbol')} (volume: {vol_weighted[0].get('volume', 0):,.0f})"
        )

        # Method 3: Surprise factor (big moves on high volume)
        print("\n🔍 Method 3: Surprise Factor (Breakout Signal)")
        surprise = self.ranking_engine.rank_by_surprise_factor(scan_results)
        comparison["rankings"]["surprise_factor"] = {
            "method": "Big moves on high volume",
            "top_5": [
                {
                    "symbol": r.get("symbol"),
                    "surprise_score": round(
                        r.get("combined_momentum", 0) * (r.get("volume", 1) ** 0.5) / 1000, 6
                    ),
                    "magnitude": round(r.get("hourly_return_pct", 0), 3),
                    "volume": r.get("volume", 0),
                }
                for r in surprise[:5]
            ],
            "bottom_5": [
                {
                    "symbol": r.get("symbol"),
                    "surprise_score": round(
                        r.get("combined_momentum", 0) * (r.get("volume", 1) ** 0.5) / 1000, 6
                    ),
                    "magnitude": round(r.get("hourly_return_pct", 0), 3),
                    "volume": r.get("volume", 0),
                }
                for r in surprise[-5:]
            ],
        }
        print(f"  Top symbol: {surprise[0].get('symbol')} (surprise: high volume + big move)")

        return comparison

    def analyze_ranking_overlap(self, scan_results: list[dict], top_n: int = 20) -> dict:
        """Analyze how many symbols appear in top-N across different ranking methods."""
        print(f"\n📈 Analyzing top-{top_n} overlap across methods...")

        standard = self.ranking_engine.rank_by_metric(scan_results, metric="combined")[:top_n]
        vol_weighted = self.ranking_engine.rank_by_volume_weighted_momentum(scan_results)[:top_n]
        surprise = self.ranking_engine.rank_by_surprise_factor(scan_results)[:top_n]

        standard_syms = {s["symbol"] for s in standard}
        vol_weighted_syms = {s["symbol"] for s in vol_weighted}
        surprise_syms = {s["symbol"] for s in surprise}

        overlap = {
            "standard_only": len(standard_syms - vol_weighted_syms - surprise_syms),
            "vol_weighted_only": len(vol_weighted_syms - standard_syms - surprise_syms),
            "surprise_only": len(surprise_syms - standard_syms - vol_weighted_syms),
            "standard_vol_overlap": len(standard_syms & vol_weighted_syms - surprise_syms),
            "standard_surprise_overlap": len(standard_syms & surprise_syms - vol_weighted_syms),
            "vol_surprise_overlap": len(vol_weighted_syms & surprise_syms - standard_syms),
            "all_three_overlap": len(standard_syms & vol_weighted_syms & surprise_syms),
            "symbols_in_all_three": list(standard_syms & vol_weighted_syms & surprise_syms),
        }

        return overlap

    def apply_filters_per_method(self, scan_results: list[dict], filter_config: dict) -> dict:
        """Apply filters and see how each ranking method is affected."""
        print(f"\n🎯 Applying filters: {filter_config}")

        filtered = self.filters.apply_all_filters(scan_results, **filter_config)
        print(f"  Symbols after filtering: {len(filtered)} / {len(scan_results)}")

        # Compare rankings on filtered results
        standard = self.ranking_engine.rank_by_metric(filtered, metric="combined")
        vol_weighted = self.ranking_engine.rank_by_volume_weighted_momentum(filtered)
        surprise = self.ranking_engine.rank_by_surprise_factor(filtered)

        return {
            "total_after_filter": len(filtered),
            "standard_top_10": [s["symbol"] for s in standard[:10]],
            "vol_weighted_top_10": [s["symbol"] for s in vol_weighted[:10]],
            "surprise_top_10": [s["symbol"] for s in surprise[:10]],
        }

    def generate_report(
        self,
        comparison: dict,
        overlap: dict,
        filtered_analysis: dict,
        output_file: str = "ranking_comparison_report.json",
    ) -> str:
        """Generate comprehensive comparison report."""
        report = {
            "metadata": {"generated_at": datetime.now().isoformat(), "timezone": "UTC"},
            "comparison": comparison,
            "overlap_analysis": overlap,
            "filtered_analysis": filtered_analysis,
            "insights": self._generate_insights(comparison, overlap),
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved to {output_file}")
        return output_file

    def _generate_insights(self, comparison: dict, overlap: dict) -> dict:
        """Generate actionable insights from comparison."""
        return {
            "consensus_signals": overlap["symbols_in_all_three"],
            "high_confidence_message": f"{overlap['all_three_overlap']} symbols appear in top rankings across ALL methods - highest confidence signals",
            "method_preferences": {
                "standard_rank_best_for": "Most balanced momentum measure",
                "volume_weighted_best_for": "High liquidity, institutional activity",
                "surprise_factor_best_for": "Breakout detection, unexpected moves",
            },
            "recommendation": "Use consensus signals (all 3 methods) for highest confidence trades",
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare momentum ranking methods")
    parser.add_argument(
        "--universe", default="sp500", help="Symbol universe (sp500, nasdaq100, all)"
    )
    parser.add_argument("--symbols", type=int, default=100, help="Number of symbols to scan")
    parser.add_argument("--data-source", default="alpaca", help="Data source (alpaca, databento)")
    parser.add_argument("--output", default="ranking_comparison_report.json", help="Output file")
    parser.add_argument("--min-volume", type=int, default=100000, help="Minimum volume filter")
    parser.add_argument(
        "--min-magnitude", type=float, default=0.005, help="Minimum magnitude filter"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = RankingComparison(data_source=args.data_source)

    # Run scan
    results = analyzer.run_scan(universe=args.universe, num_symbols=args.symbols)

    if not results:
        print("❌ No scan results. Check credentials and data source.")
        return

    # Compare ranking methods
    comparison = analyzer.compare_rankings(results)

    # Analyze overlap
    overlap = analyzer.analyze_ranking_overlap(results, top_n=20)
    print("\n📊 Top-20 Overlap:")
    print(f"  All 3 methods: {overlap['all_three_overlap']} symbols")
    print(f"  Consensus signals: {overlap['symbols_in_all_three']}")

    # Apply filters
    filter_config = {"min_volume": args.min_volume, "min_magnitude": args.min_magnitude}
    filtered_analysis = analyzer.apply_filters_per_method(results, filter_config)

    # Generate report
    analyzer.generate_report(comparison, overlap, filtered_analysis, output_file=args.output)

    print(
        f"\n🎯 Key Insight: {overlap['all_three_overlap']} high-confidence signals (in all 3 ranking methods)"
    )


if __name__ == "__main__":
    main()
