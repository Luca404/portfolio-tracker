# backend/etf_cache.py
"""
ETF Cache System - Lista statica degli ETF più comuni
Questa lista contiene gli ETF più popolari e viene aggiornata manualmente
"""

ETF_CACHE = [
    # US Equity - Broad Market
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "IVV", "name": "iShares Core S&P 500 ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # US Equity - Growth/Value
    {"symbol": "VUG", "name": "Vanguard Growth ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VTV", "name": "Vanguard Value ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "SCHG", "name": "Schwab U.S. Large-Cap Growth ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "SCHV", "name": "Schwab U.S. Large-Cap Value ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # International Equity
    {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VWO", "name": "Vanguard FTSE Emerging Markets ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "IEFA", "name": "iShares Core MSCI EAFE ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VXUS", "name": "Vanguard Total International Stock ETF", "exchange": "NASDAQ", "currency": "USD"},
    
    # Bonds
    {"symbol": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "LQD", "name": "iShares iBoxx Investment Grade Corporate Bond ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "HYG", "name": "iShares iBoxx High Yield Corporate Bond ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "TIP", "name": "iShares TIPS Bond ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # Sector ETFs
    {"symbol": "XLK", "name": "Technology Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLF", "name": "Financial Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLE", "name": "Energy Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLV", "name": "Health Care Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLP", "name": "Consumer Staples Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLY", "name": "Consumer Discretionary Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLI", "name": "Industrial Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLB", "name": "Materials Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLU", "name": "Utilities Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "XLRE", "name": "Real Estate Select Sector SPDR Fund", "exchange": "NYSE Arca", "currency": "USD"},
    
    # Thematic/Specialty
    {"symbol": "ARK", "name": "ARK Innovation ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "ARKK", "name": "ARK Innovation ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "ARKW", "name": "ARK Next Generation Internet ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "ARKG", "name": "ARK Genomic Revolution ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VNQ", "name": "Vanguard Real Estate ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "GLD", "name": "SPDR Gold Shares", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "SLV", "name": "iShares Silver Trust", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "USO", "name": "United States Oil Fund", "exchange": "NYSE Arca", "currency": "USD"},
    
    # ESG/Sustainable
    {"symbol": "ESGU", "name": "iShares ESG Aware MSCI USA ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "ESGV", "name": "Vanguard ESG U.S. Stock ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "SUSL", "name": "iShares ESG MSCI USA Leaders ETF", "exchange": "NASDAQ", "currency": "USD"},
    
    # Small/Mid Cap
    {"symbol": "IJH", "name": "iShares Core S&P Mid-Cap ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "IJR", "name": "iShares Core S&P Small-Cap ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VO", "name": "Vanguard Mid-Cap ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VB", "name": "Vanguard Small-Cap ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # Dividend
    {"symbol": "VYM", "name": "Vanguard High Dividend Yield ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "DVY", "name": "iShares Select Dividend ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "SCHD", "name": "Schwab U.S. Dividend Equity ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "SDY", "name": "SPDR S&P Dividend ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # European ETFs
    {"symbol": "VEUR", "name": "Vanguard FTSE Europe ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "VGK", "name": "Vanguard FTSE Europe ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "EZU", "name": "iShares MSCI Eurozone ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # Asia/Pacific
    {"symbol": "VPL", "name": "Vanguard FTSE Pacific ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "EWJ", "name": "iShares MSCI Japan ETF", "exchange": "NYSE Arca", "currency": "USD"},
    {"symbol": "MCHI", "name": "iShares MSCI China ETF", "exchange": "NASDAQ", "currency": "USD"},
    {"symbol": "FXI", "name": "iShares China Large-Cap ETF", "exchange": "NYSE Arca", "currency": "USD"},
    
    # Canada
    {"symbol": "EWC", "name": "iShares MSCI Canada ETF", "exchange": "NYSE Arca", "currency": "USD"},
]


def search_etf_cache(query: str, limit: int = 20):
    """
    Cerca nella cache locale degli ETF
    
    Args:
        query: termine di ricerca (symbol o name)
        limit: numero massimo di risultati
        
    Returns:
        Lista di ETF che matchano la query
    """
    query = query.upper().strip()
    results = []
    
    # Cerca per symbol (match esatto ha priorità)
    exact_match = [etf for etf in ETF_CACHE if etf["symbol"].upper() == query]
    if exact_match:
        results.extend(exact_match)
    
    # Cerca per symbol che inizia con query
    starts_with = [etf for etf in ETF_CACHE if etf["symbol"].upper().startswith(query) and etf not in results]
    results.extend(starts_with)
    
    # Cerca per symbol che contiene query
    contains_symbol = [etf for etf in ETF_CACHE if query in etf["symbol"].upper() and etf not in results]
    results.extend(contains_symbol)
    
    # Cerca nel nome
    contains_name = [etf for etf in ETF_CACHE if query in etf["name"].upper() and etf not in results]
    results.extend(contains_name)
    
    return results[:limit]


def get_all_etfs():
    """Restituisce tutti gli ETF nella cache"""
    return ETF_CACHE.copy()