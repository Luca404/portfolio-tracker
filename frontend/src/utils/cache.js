/**
 * Invalidate portfolio cache in sessionStorage
 * @param {number|string} portfolioId - Portfolio ID to invalidate
 */
export function invalidatePortfolioCache(portfolioId) {
  if (!portfolioId) return;
  const cacheKey = `portfolio_${portfolioId}`;
  sessionStorage.removeItem(cacheKey);
  console.log(`[CACHE] Portfolio ${portfolioId}: cache invalidata`);
}
