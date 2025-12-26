/**
 * Parse a date string in DD/MM/YYYY or DD-MM-YYYY format
 * @param {string} value - Date string to parse
 * @returns {Date|null} Parsed date or null if invalid
 */
export function parseDateDMY(value) {
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
}

/**
 * Convert DD/MM/YYYY or DD-MM-YYYY to ISO date format (YYYY-MM-DD)
 * @param {string} value - Date string to convert
 * @returns {string} ISO formatted date string or empty string if invalid
 */
export function toISODateFromDMY(value) {
  const d = parseDateDMY(value);
  if (!d || Number.isNaN(d.getTime())) return '';
  return d.toISOString().split('T')[0];
}
