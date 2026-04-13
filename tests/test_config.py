"""Unit tests for config/settings.yaml integrity."""
import os

import pytest
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


CORE_UNIVERSE = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'SHY', 'XLU', 'XLP']


class TestUniverseConfig:
    def test_core_symbols_present(self, config):
        # Universe is in assets.equities.universe
        universe = config.get('assets', {}).get('equities', {}).get('universe', [])
        crypto = config.get('assets', {}).get('crypto', {}).get('universe', [])
        all_syms = universe + crypto
        for sym in CORE_UNIVERSE:
            assert sym in all_syms, f"{sym} missing from universe config"

    def test_crypto_present(self, config):
        crypto = config.get('assets', {}).get('crypto', {}).get('universe', [])
        has_btc = any('BTC' in str(s) for s in crypto)
        has_eth = any('ETH' in str(s) for s in crypto)
        assert has_btc, "BTC missing from universe"
        assert has_eth, "ETH missing from universe"


class TestRebalanceConfig:
    def test_adaptive_rebalance(self, config):
        freq = config.get('strategy', {}).get('rebalance_frequency')
        assert freq == 'adaptive', f"rebalance_frequency should be 'adaptive', got {freq}"


class TestExecutionConfig:
    def test_hourly_timing_enabled(self, config):
        exec_conf = config.get('execution', {})
        assert exec_conf.get('hourly_timing_enabled')

    def test_dynamic_universe_enabled(self, config):
        exec_conf = config.get('execution', {})
        assert exec_conf.get('dynamic_universe_enabled')


class TestWeightVectors:
    def _get_weights(self, config, key):
        regime = config.get('strategy', {}).get('regime_switching', {})
        return regime.get(key, {})

    def test_no_duplicate_top_level_keys(self):
        """PyYAML silently overwrites duplicates — detect them via raw text parse."""
        with open(CONFIG_PATH) as f:
            content = f.read()
        top_keys = [line.split(':')[0].strip() for line in content.split('\n')
                    if line and not line.startswith(' ') and ':' in line and not line.startswith('#')]
        duplicates = [k for k in set(top_keys) if top_keys.count(k) > 1]
        assert len(duplicates) == 0, f"Duplicate top-level YAML keys: {duplicates}"

    def test_strategy_has_all_keys(self, config):
        """Strategy block should contain all keys from both original blocks."""
        strategy = config.get('strategy', {})
        assert 'name' in strategy, "strategy.name missing"
        assert 'rebalance_frequency' in strategy, "strategy.rebalance_frequency missing"
        assert 'regime_switching' in strategy, "strategy.regime_switching missing"
        assert 'lookback_fast' in strategy, "strategy.lookback_fast missing"
