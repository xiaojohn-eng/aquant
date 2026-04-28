"""WFA Engine patch for _run_single_period bug fix.

Original bug: pct_change() on multi-stock DataFrame computed cross-sectional
changes instead of time-series returns, causing astronomical return values.

Fix: Compute holding-period returns from first-day buy to last-day sell.
"""
