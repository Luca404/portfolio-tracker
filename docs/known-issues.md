# portfolio-tracker — Known Issues & TODO

## Known issues

**ETF intraday prices**: yfinance and JustETF provide EOD prices only. European ETF prices shown during the day are last close. No reliable intraday source for UCITS ETFs on XETRA/Euronext. Possible direction: `yf.Ticker.fast_info.last_price` as separate current price field, but coverage for European ETFs is limited.

**ETF UCITS cache — monthly refresh**: table was seeded with 2834 ETFs. New ETFs auto-added on-demand via `/symbols/isin-lookup`. For systematic refresh: re-run `scripts/scrape_justetf.py` + `scripts/generate_etf_cache.py` + re-import CSV to Supabase (~monthly).

**Portfolio history date gaps**: `aggregate_portfolio_history()` in `utils/portfolio.py` skips dates where any active asset has no price data. Forward-fill is implemented but doesn't help when yfinance returns insufficient history for an ETF (e.g. limited listing history). To debug: identify which symbol has limited history, then decide whether to exclude it from gap dates instead of invalidating the entire date.

**Bond portfolio math**: bonds are saved with `price` as % of nominal. Portfolio aggregation should use `nominal × price / 100` instead of `quantity × price`. `utils/portfolio.py` and `utils/pricing.py` don't handle this yet — defer until full bond analytics support is added.

**US Treasuries and corporate bonds**: not yet supported. Possible sources: EODHD (free tier) for US Treasuries, TreasuryDirect API, EODHD or Borsa Italiana scraping for corporate bonds.

**GBP risk-free rate**: hardcoded at 4.0%. Should use UK Gilt yield from a public API.

**`history_mode` not consumed by analytics**: field is stored (`full_orders`/`positions_only`) but analytics don't gate on it yet. XIRR and other full-history metrics should be hidden for `positions_only` portfolios.

**Multi-profile support**: pfTrackr creates portfolios with `profile_id = user_id` (main profile only). Full multi-profile support in pfTrackr backend/frontend not yet implemented — see `future-improvements.local` item #3 in trackr/.
