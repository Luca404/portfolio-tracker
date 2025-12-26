/**
 * Get currency symbol from currency code
 * @param {string} currency - Currency code (e.g., 'USD', 'EUR')
 * @returns {string} Currency symbol
 */
export function getCurrencySymbol(currency) {
  const symbols = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CHF': 'CHF',
    'CAD': 'C$',
    'AUD': 'A$',
    'CNY': '¥'
  };
  return symbols[currency] || currency;
}

/**
 * Format a numeric value as currency
 * @param {number} val - Value to format
 * @param {string} currency - Currency code
 * @returns {string} Formatted currency string
 */
export function formatCurrencyValue(val, currency) {
  const symbol = getCurrencySymbol(currency);
  const num = Number(val || 0);
  return `${symbol}${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/**
 * Format TER (Total Expense Ratio) value
 * @param {string|number} val - TER value
 * @returns {string} Formatted TER string with % symbol
 */
export function formatTerValue(val) {
  if (val === null || val === undefined) return '—';
  const strVal = val.toString().trim();
  if (!strVal) return '—';
  if (strVal.endsWith('%')) return strVal;
  const num = Number(strVal);
  return Number.isNaN(num) ? `${strVal}%` : `${num}%`;
}
