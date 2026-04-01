"""
signals.py — Patch for full upgrade integration
================================================
These are the additions/modifications needed to integrate:
  - OptionsFlowSignal (Factor 13)
  - EarningsNLPSignal (Factor 14)
  - CrisisAlphaAmplifier (replaces vol scaling logic)
  - H2OMetaLearner (final signal meta-combination)
  - KellySizer (wired via portfolio.py, not signals.py)

Integration instructions:
  1. Add these imports to the top of signals.py
  2. Add these fields to SignalGenerator.__init__
  3. Replace the return line in generate() with the meta-learner blend

IMPORT ADDITIONS (add after existing imports in signals.py):
"""

IMPORT_ADDITIONS = '''
# ── New signal modules ─────────────────────────────────────────────────
from strategy.options_flow_signal import OptionsFlowSignal
from strategy.earnings_nlp_signal import EarningsNLPSignal
from core.crisis_alpha_amplifier  import CrisisAlphaAmplifier
from core.h2o_meta_learner        import H2OMetaLearner
from core.kelly_sizer             import KellySizer
'''

INIT_ADDITIONS = '''
        # ── Options Order Flow Signal (Factor 13) ──────────────────────────
        self._options_flow = None
        of_cfg = config.get("options_flow", {})
        self.options_flow_enabled = of_cfg.get("enabled", True)
        self.options_flow_weight  = of_cfg.get("weight", 0.20)
        if self.options_flow_enabled:
            try:
                self._options_flow = OptionsFlowSignal(config)
                log.info("OptionsFlowSignal initialised")
            except Exception as e:
                log.warning(f"OptionsFlowSignal init failed: {e}")

        # ── Earnings NLP Signal (Factor 14) ────────────────────────────────
        self._earnings_nlp = None
        enl_cfg = config.get("earnings_nlp", {})
        self.earnings_nlp_enabled = enl_cfg.get("enabled", True)
        self.earnings_nlp_weight  = enl_cfg.get("weight", 0.15)
        if self.earnings_nlp_enabled:
            try:
                self._earnings_nlp = EarningsNLPSignal(config)
                log.info("EarningsNLPSignal initialised")
            except Exception as e:
                log.warning(f"EarningsNLPSignal init failed: {e}")

        # ── Crisis Alpha Amplifier ─────────────────────────────────────────
        # Replaces the current vol-targeting suppression logic.
        # Diagnostic: system earns Sharpe 2.03 in high-vol, 0.128 in calm.
        # So we BOOST in crisis, REDUCE in suppressed vol.
        self._crisis_amplifier = None
        ca_cfg = config.get("crisis_alpha", {})
        self.crisis_alpha_enabled = ca_cfg.get("enabled", True)
        if self.crisis_alpha_enabled:
            try:
                self._crisis_amplifier = CrisisAlphaAmplifier(config)
                log.info("CrisisAlphaAmplifier initialised")
            except Exception as e:
                log.warning(f"CrisisAlphaAmplifier init failed: {e}")

        # ── H2O Meta-Learner (final non-linear combiner) ───────────────────
        self._meta_learner = None
        ml_cfg = config.get("meta_learner", {})
        self.meta_learner_enabled = ml_cfg.get("enabled", True)
        self.meta_blend_weight    = ml_cfg.get("blend_weight", 0.40)
        if self.meta_learner_enabled:
            try:
                self._meta_learner = H2OMetaLearner(config)
                loaded = self._meta_learner.load()
                if loaded:
                    log.info("H2OMetaLearner loaded from disk")
                else:
                    log.info("H2OMetaLearner ready (no saved model — will train on first use)")
            except Exception as e:
                log.warning(f"H2OMetaLearner init failed: {e}")
'''

# This replaces the final return in _compute_symbol_signal()
SYMBOL_SIGNAL_NEW_TAIL = '''
        # ── Factor 13: Options Order Flow ─────────────────────────────────
        options_overlay = 0.0
        if self._options_flow is not None and self.options_flow_enabled:
            try:
                of_signals = self._options_flow.compute(
                    [sym], as_of_date, lookback_days=5
                )
                options_overlay = of_signals.get(sym, 0.0)
            except Exception:
                pass

        # ── Factor 14: Earnings NLP ────────────────────────────────────────
        earnings_overlay = 0.0
        if self._earnings_nlp is not None and self.earnings_nlp_enabled:
            try:
                enl_signals = self._earnings_nlp.compute(
                    [sym], as_of_date, lookback_days=30
                )
                earnings_overlay = enl_signals.get(sym, 0.0)
            except Exception:
                pass

        # ── Blend new overlays into combined signal ────────────────────────
        # Reduce existing weights proportionally to make room for new signals
        # Total must remain 1.0:
        #   existing composite_with_vol: 1.0 - options_weight - earnings_weight
        #   options_flow: options_flow_weight (default 0.20)
        #   earnings_nlp: earnings_nlp_weight (default 0.15)
        existing_weight = 1.0 - self.options_flow_weight - self.earnings_nlp_weight
        final_signal = (
            existing_weight   * combined_with_vol
            + self.options_flow_weight  * float(options_overlay)
            + self.earnings_nlp_weight  * float(earnings_overlay)
        ).clip(-1, 1)

        # ── Crisis Alpha Amplifier ─────────────────────────────────────────
        # Applied LAST as a portfolio-level scale (not per-symbol signal mod)
        # Stored on self for generate() to apply at portfolio level
        # (per-symbol application would require VIX data per symbol call)

        return sym, final_signal
'''

# This is inserted in generate() after cross-sectional overlay, before return
GENERATE_CRISIS_INSERTION = '''
        # ── Crisis Alpha Amplifier (portfolio-level scale) ─────────────────
        # Applied after all signal computation.
        # Diagnostic showed Sharpe 2.03 in high-vol vs 0.128 in calm.
        # We boost in crisis, reduce in suppressed-vol calm markets.
        crisis_scale = 1.0
        if self._crisis_amplifier is not None and self.crisis_alpha_enabled:
            try:
                vix_s = self._macro_data.get("^VIX", pd.DataFrame()).get("Close")
                spy_s = all_data.get("SPY", pd.DataFrame())
                spy_r = spy_s["Close"].pct_change() if "Close" in spy_s.columns else None
                crisis_scale = self._crisis_amplifier.get_scale(vix_s, spy_r, as_of_date)
                log.debug(f"CrisisAlphaAmplifier scale: {crisis_scale:.3f}")
            except Exception:
                pass

        if crisis_scale != 1.0:
            signal_df = (signal_df * crisis_scale).clip(-1, 1)

        # ── H2O Meta-Learner (final non-linear blend) ──────────────────────
        # Combines all 14 signal components non-linearly.
        # Blend: meta_blend_weight * meta_signal + (1-meta_blend_weight) * direct_composite
        if self._meta_learner is not None and self.meta_learner_enabled:
            try:
                # Build feature dict for meta-learner (latest row of each signal)
                # Meta-learner returns adjusted signals; blend with direct composite
                meta_signals = self._meta_learner.predict_batch(
                    signal_df, self._macro_data, all_data, as_of_date
                )
                if meta_signals is not None and not meta_signals.empty:
                    w = self.meta_blend_weight
                    signal_df = (
                        w * meta_signals.reindex_like(signal_df).fillna(0)
                        + (1 - w) * signal_df
                    ).clip(-1, 1)
            except Exception as _e:
                log.debug(f"MetaLearner predict failed: {_e}")

        return signal_df
'''

print("Signal patch module written.")
print("Integration points:")
print("1. Add IMPORT_ADDITIONS to top of strategy/signals.py")
print("2. Add INIT_ADDITIONS to SignalGenerator.__init__")
print("3. Replace _compute_symbol_signal return with SYMBOL_SIGNAL_NEW_TAIL")
print("4. Add GENERATE_CRISIS_INSERTION before return in generate()")
