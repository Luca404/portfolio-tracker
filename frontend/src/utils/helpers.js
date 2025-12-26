// Helper per invalidare cache portfolio
export const invalidatePortfolioCache = (portfolioId) => {
  if (!portfolioId) return;
  const cacheKey = `portfolio_${portfolioId}`;
  sessionStorage.removeItem(cacheKey);
  console.log(`[CACHE] Portfolio ${portfolioId}: cache invalidata`);
};

export const formatCurrencyValue = (val, currency) => {
  const getCurrencySymbol = (curr) => {
    switch(curr) {
      case 'EUR': return '€';
      case 'USD': return '$';
      case 'GBP': return '£';
      case 'CHF': return 'Fr';
      case 'JPY': return '¥';
      case 'CNY': return '¥';
      default: return curr ? `${curr} ` : '';
    }
  };
  const symbol = getCurrencySymbol(currency);
  const num = Number(val || 0);
  return `${symbol}${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

export const formatTerValue = (val) => {
  if (val === null || val === undefined) return '—';
  const strVal = val.toString().trim();
  if (!strVal) return '—';
  if (strVal.endsWith('%')) return strVal;
  const num = Number(strVal);
  return Number.isNaN(num) ? `${strVal}%` : `${num}%`;
};

export const parseDateDMY = (value) => {
  if (!value) return null;
  const parts = value.split(/[-/]/).map((p) => parseInt(p, 10));
  if (parts.length !== 3 || parts.some(isNaN)) return null;
  let day, month, year;
  if (parts[0] > 999) {
    [year, month, day] = parts;
  } else {
    [day, month, year] = parts;
  }
  return new Date(Date.UTC(year, month - 1, day));
};

export const toISODateFromDMY = (value) => {
  const d = parseDateDMY(value);
  if (!d || Number.isNaN(d.getTime())) return '';
  return d.toISOString().split('T')[0];
};

export function getCurrencySymbol(currency) {
  const symbols = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CHF': 'CHF',
    'CAD': 'C$',
    'AUD': 'A$'
  };
  return symbols[currency] || currency;
}
