import React, { useState, useEffect } from 'react';
import { Plus, RefreshCw, Edit2, Trash2 } from 'lucide-react';
import { API_URL } from '../config';
import { formatCurrencyValue, formatTerValue, parseDateDMY, toISODateFromDMY, invalidatePortfolioCache } from '../utils/helpers';

function OrdersPage({ token, portfolio, portfolios, onSelectPortfolio, refreshPortfolios }) {
  const [orders, setOrders] = useState([]);
  const [ordersLoading, setOrdersLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingOrderId, setEditingOrderId] = useState(null);
  const [symbolOptions, setSymbolOptions] = useState([]);
  const [symbolLoading, setSymbolLoading] = useState(false);
  const [searchCompleted, setSearchCompleted] = useState(false);
  const [ucitsCache, setUcitsCache] = useState([]);
  const [selectedInfo, setSelectedInfo] = useState({ name: '', exchange: '', currency: '', ter: '', isin: '' });
  const [skipSearch, setSkipSearch] = useState(false);
  const [lastChosenSymbol, setLastChosenSymbol] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [ordersPerPage, setOrdersPerPage] = useState(10);
  const [submitting, setSubmitting] = useState(false);
  const [isinLookupLoading, setIsinLookupLoading] = useState(false);
  const [isinLookupError, setIsinLookupError] = useState(false);
  const [bondMeta, setBondMeta] = useState(null);
  const [bondLookupLoading, setBondLookupLoading] = useState(false);
  const [bondLookupError, setBondLookupError] = useState(false);
  const formRef = React.useRef(null);
  const scrollToForm = () => {
    requestAnimationFrame(() => {
      if (formRef.current) {
        formRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  };

  const getCurrencySymbol = (curr) => {
    switch(curr) {
      case 'EUR': return '€';
      case 'USD': return '$';
      case 'GBP': return '£';
      case 'CHF': return 'Fr';
      case 'JPY': return '¥';
      case 'CNY': return '¥';
      default: return '';
    }
  };
  const currencySymbol = getCurrencySymbol(selectedInfo.currency);

  const [formData, setFormData] = useState({
    symbol: '',
    quantity: '',
    price: '',
    commission: '0',
    instrument_type: 'etf',
    order_type: 'buy',
    date: new Date().toISOString().split('T')[0]
  });
  const [touched, setTouched] = useState({});

  // Flag per evitare fetch multipli
  const fetchingOrdersRef = React.useRef(false);

  useEffect(() => {
    const loadOrders = async () => {
      if (fetchingOrdersRef.current) return;

      fetchingOrdersRef.current = true;
      try {
        await fetchOrders();
      } finally {
        fetchingOrdersRef.current = false;
      }
    };

    loadOrders();
  }, [portfolio.id]); // Dipende solo da ID, non oggetto intero

  const handlePortfolioChange = (e) => {
    const next = portfolios.find(p => p.id === Number(e.target.value));
    if (next) {
      onSelectPortfolio(next);
    }
  };

  const fetchOrders = async () => {
    setOrdersLoading(true);
    try {
      const res = await fetch(`${API_URL}/orders/${portfolio.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      const sorted = (data.orders || []).slice().sort((a, b) => {
        const da = parseDateDMY(a.date) || new Date(a.date);
        const db = parseDateDMY(b.date) || new Date(b.date);
        return db - da;
      });
      setOrders(sorted);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setOrdersLoading(false);
    }
  };

  // Flag per evitare fetch multipli UCITS
  const ucitsLoadedRef = React.useRef(false);

  useEffect(() => {
    const loadUcits = async () => {
      if (ucitsLoadedRef.current || ucitsCache.length > 0) return;

      // Controlla cache in sessionStorage (valida per sessione)
      const cached = sessionStorage.getItem('ucits_etf_list');
      if (cached) {
        try {
          const data = JSON.parse(cached);
          setUcitsCache(data);
          console.log(`[CACHE] UCITS from cache: ${data.length} ETFs`);
          return;
        } catch (e) {
          console.warn('[CACHE] UCITS cache error:', e);
        }
      }

      ucitsLoadedRef.current = true;
      try {
        const res = await fetch(`${API_URL}/symbols/ucits`);
        if (res.ok) {
          const data = await res.json();
          const ucitsList = data.results || [];
          setUcitsCache(ucitsList);

          // Salva in cache
          try {
            sessionStorage.setItem('ucits_etf_list', JSON.stringify(ucitsList));
            console.log(`[CACHE] UCITS loaded & cached: ${ucitsList.length} ETFs`);
          } catch (e) {
            console.warn('[CACHE] UCITS save error:', e);
          }
        }
      } catch (e) {
        console.error('UCITS cache load error', e);
        ucitsLoadedRef.current = false; // Reset on error
      }
    };
    loadUcits();
  }, []); // Nessuna dipendenza - carica solo una volta

  const parseNum = (val) => parseFloat(String(val).replace(',', '.'));
  const isIsin = (s) => /^[A-Z]{2}[A-Z0-9]{10}$/.test(s);

  const handleBondLookup = async () => {
    setBondLookupLoading(true);
    setBondLookupError(false);
    setBondMeta(null);
    try {
      const res = await fetch(`${API_URL}/symbols/bond-lookup?isin=${formData.symbol}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('not found');
      const data = await res.json();
      const meta = data.metadata || {};
      setBondMeta(meta);
      setSelectedInfo({
        name: meta.name || meta.issuer || '',
        exchange: 'MOT/EuroMOT',
        currency: meta.currency || 'EUR',
        ter: '',
        isin: formData.symbol,
      });
      setLastChosenSymbol(formData.symbol);
    } catch {
      setBondLookupError(true);
    } finally {
      setBondLookupLoading(false);
    }
  };

  const handleIsinLookup = async () => {
    setIsinLookupLoading(true);
    setIsinLookupError(false);
    try {
      const res = await fetch(`${API_URL}/symbols/isin-lookup?isin=${formData.symbol}`);
      if (!res.ok) throw new Error('not found');
      const data = await res.json();
      const entries = data.listings.map(l => ({
        symbol: l.ticker,
        isin: formData.symbol,
        name: l.name,
        exchange: l.exchange,
        currency: l.currency,
        ter: l.ter,
      }));
      setUcitsCache(prev => [...prev, ...entries]);
      setSymbolOptions(entries);
      setSearchCompleted(true);
    } catch {
      setIsinLookupError(true);
    } finally {
      setIsinLookupLoading(false);
    }
  };

  const handleSubmit = async () => {
    const errs = {};
    const qtyVal = parseNum(formData.quantity);
    const priceVal = parseNum(formData.price);
    const symbolIsIsin = isIsin(formData.symbol);
    if (!formData.symbol || (!selectedInfo.name && !symbolIsIsin)) errs.symbol = true;
    if (!formData.quantity || isNaN(qtyVal) || qtyVal <= 0) errs.quantity = true;
    if (!formData.price || isNaN(priceVal) || priceVal <= 0) errs.price = true;
    if (Object.keys(errs).length > 0) {
      setTouched({...touched, ...errs});
      return;
    }
    setSubmitting(true);
    try {
      const datePayload = toISODateFromDMY(formData.date) || formData.date;
      const symbolIsIsin = isIsin(formData.symbol);
      const payload = {
        portfolio_id: portfolio.id,
        symbol: symbolIsIsin ? (selectedInfo.isin || formData.symbol) : formData.symbol,
        isin: selectedInfo.isin || (symbolIsIsin ? formData.symbol : ''),
        name: selectedInfo.name,
        exchange: selectedInfo.exchange,
        currency: selectedInfo.currency,
        ter: selectedInfo.ter,
        quantity: qtyVal,
        price: priceVal,
        commission: parseNum(formData.commission || 0),
        instrument_type: formData.instrument_type,
        order_type: formData.order_type,
        date: datePayload
      };

      const url = editingOrderId ? `${API_URL}/orders/${editingOrderId}` : `${API_URL}/orders`;
      const method = editingOrderId ? 'PUT' : 'POST';

      const res = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setShowForm(false);
        setFormData({
          symbol: '', quantity: '', price: '', commission: '0', instrument_type: 'etf', order_type: 'buy',
          date: new Date().toISOString().split('T')[0]
        });
        setSelectedInfo({ name: '', exchange: '', currency: '', ter: '' });
        setLastChosenSymbol('');
        setTouched({});
        setEditingOrderId(null);
        setSymbolOptions([]);
        setSearchCompleted(false);
        await fetchOrders();
        // Invalida cache quando vengono modificati ordini
        invalidatePortfolioCache(portfolio?.id);
        refreshPortfolios();
      }
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  useEffect(() => {
    if (skipSearch) {
      setSkipSearch(false);
      return;
    }
    if (formData.symbol && formData.symbol === lastChosenSymbol) {
      return;
    }

    // Reset stato quando l'utente cambia input
    setSearchCompleted(false);

    const controller = new AbortController();
    const fetchSuggestions = async () => {
      if (!formData.symbol || formData.symbol.length < 2) {
        setSymbolOptions([]);
        setSelectedInfo({ name: '', exchange: '', currency: '' });
        setSymbolLoading(false);
        setSearchCompleted(false);
        return;
      }

      setSymbolLoading(true);

      // Bond: no ticker search — use ISIN lookup only
      if (formData.instrument_type === 'bond') {
        setSymbolOptions([]);
        setSymbolLoading(false);
        setSearchCompleted(false);
        return;
      }

      // local search for etf using ucits cache
      if (formData.instrument_type === 'etf') {
        // Simula un breve delay per mostrare il loading
        await new Promise(resolve => setTimeout(resolve, 100));
        const q = formData.symbol.toUpperCase();
        const filtered = ucitsCache
          .filter(item => {
            const ticker = (item.symbol || '').toUpperCase();
            const isin = (item.isin || '').toUpperCase();
            const tickerMatch = ticker.startsWith(q);
            const isinMatch = q.length === 12 && isin === q;
            return tickerMatch || isinMatch;
          })
          .slice(0, 25);
        setSymbolOptions(filtered);
        setSymbolLoading(false);
        setSearchCompleted(true);
        return;
      }

      try {
        const res = await fetch(
          `${API_URL}/symbols/search?q=${encodeURIComponent(formData.symbol)}&instrument_type=${formData.instrument_type}`,
          { signal: controller.signal }
        );
        if (res.ok) {
          const data = await res.json();
          setSymbolOptions(data.results || []);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Symbol search error:', err);
        }
      } finally {
        setSymbolLoading(false);
        setSearchCompleted(true);
      }
    };
    const timer = setTimeout(fetchSuggestions, 250);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [formData.symbol, formData.instrument_type, skipSearch, ucitsCache]);


  const handleDeleteOrder = async () => {
    if (!deleteTarget) return;
    try {
      await fetch(`${API_URL}/orders/${deleteTarget.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      await fetchOrders();
      // Invalida cache quando viene cancellato un ordine
      invalidatePortfolioCache(portfolio?.id);
      refreshPortfolios();
    } catch (err) {
      console.error('Delete error:', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Orders</h1>
          <p className="text-slate-600">Manage portfolio transactions</p>
          <div className="mt-2">
            <select
              value={portfolio.id}
              onChange={handlePortfolioChange}
              className="px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          onClick={() => {
            if (showForm) {
              // Chiude il form
              setShowForm(false);
              setEditingOrderId(null);
            } else {
              // Apre il form per nuovo ordine
              setShowForm(true);
              setEditingOrderId(null);
              setFormData({
                symbol: '', quantity: '', price: '', commission: '0', instrument_type: 'etf', order_type: 'buy',
                date: new Date().toISOString().split('T')[0]
              });
              setSelectedInfo({ name: '', exchange: '', currency: '' });
              setLastChosenSymbol('');
              setSearchCompleted(false);
              scrollToForm();
            }
          }}
          className="flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
        >
          <Plus className="w-5 h-5" />
          New Order
        </button>
      </div>

      {showForm && (
        <div ref={formRef} className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">{editingOrderId ? 'Edit Order' : 'Create Order'}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="relative">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                {formData.instrument_type === 'etf' ? 'Ticker or ISIN' : formData.instrument_type === 'bond' ? 'ISIN' : 'Ticker or Name'}
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => {
                    setFormData({...formData, symbol: e.target.value.toUpperCase()});
                    setSelectedInfo({ name: '', exchange: '', currency: '', ter: '', isin: '' });
                    setIsinLookupError(false);
                    setBondMeta(null);
                    setBondLookupError(false);
                  }}
                  className={`w-full px-4 py-2 ${symbolLoading ? 'pr-10' : ''} border ${touched.symbol && (!formData.symbol || (!selectedInfo.name && !isIsin(formData.symbol))) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
                  placeholder={formData.instrument_type === 'etf' ? 'Es: VWCE, SWDA' : formData.instrument_type === 'bond' ? 'Es: IT0005398406' : 'Es: AAPL, MSFT'}
                />
                {symbolLoading && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <RefreshCw size={16} className="animate-spin text-slate-400" />
                  </div>
                )}
              </div>
              {formData.symbol && formData.symbol.length >= 2 && !symbolLoading && searchCompleted && (
                <div className="absolute z-10 mt-1 border border-slate-200 rounded-lg max-h-48 overflow-auto bg-white shadow w-full">
                  {symbolOptions.length > 0 ? (
                    symbolOptions.map((opt) => (
                      <button
                        key={`${opt.symbol}-${opt.exchange || ''}`}
                        type="button"
                        onClick={() => {
                          setFormData({...formData, symbol: opt.symbol.toUpperCase()});
                          setSelectedInfo({
                            name: opt.name || '',
                            exchange: opt.exchange || '',
                            currency: opt.currency || '',
                            ter: opt.ter || '',
                            isin: opt.isin || ''
                          });
                          setSymbolOptions([]);
                          setSearchCompleted(false);
                          setSkipSearch(true);
                          setLastChosenSymbol(opt.symbol.toUpperCase());
                        }}
                        className="w-full text-left px-3 py-2 hover:bg-slate-100 text-sm"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex flex-col">
                            <span className="font-semibold text-slate-900">{opt.symbol}</span>
                            {opt.name && <span className="text-slate-600 text-xs">{opt.name}</span>}
                          </div>
                          <div className="text-xs text-slate-500 text-right">
                            {opt.exchange && <div>{opt.exchange}</div>}
                            {opt.currency && <div>{opt.currency}</div>}
                          </div>
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="px-3 py-3 text-sm text-slate-500 text-center">
                      {isIsin(formData.symbol) ? (
                        <div className="flex flex-col items-center gap-2">
                          <span>ISIN not in local cache</span>
                          {isinLookupError && <span className="text-red-500 text-xs">Not found on JustETF</span>}
                          <button
                            type="button"
                            onClick={handleIsinLookup}
                            disabled={isinLookupLoading}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white text-xs rounded-lg hover:bg-blue-700 disabled:opacity-60 transition"
                          >
                            {isinLookupLoading ? (
                              <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                              </svg>
                            ) : null}
                            {isinLookupLoading ? 'Searching...' : 'Look up on JustETF'}
                          </button>
                        </div>
                      ) : 'No results'}
                    </div>
                  )}
                </div>
              )}
              {formData.instrument_type === 'bond' && isIsin(formData.symbol) && (
                <div className="mt-2 flex flex-col items-start gap-1">
                  {bondLookupError && <span className="text-red-500 text-xs">Bond not found (Börse Frankfurt)</span>}
                  {!bondMeta && (
                    <button
                      type="button"
                      onClick={handleBondLookup}
                      disabled={bondLookupLoading}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white text-xs rounded-lg hover:bg-blue-700 disabled:opacity-60 transition"
                    >
                      {bondLookupLoading ? (
                        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                        </svg>
                      ) : null}
                      {bondLookupLoading ? 'Searching...' : 'Cerca obbligazione'}
                    </button>
                  )}
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Name</label>
                  <input
                    type="text"
                    value={selectedInfo.name}
                    readOnly
                    className="w-full px-3 py-2 border border-slate-200 rounded bg-slate-50 text-sm cursor-default select-none pointer-events-none"
                    placeholder="—"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Exchange</label>
                  <input
                    type="text"
                    value={selectedInfo.exchange}
                    readOnly
                    className="w-full px-3 py-2 border border-slate-200 rounded bg-slate-50 text-sm cursor-default select-none pointer-events-none"
                    placeholder="—"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Currency</label>
                  <select
                    value={selectedInfo.currency || ''}
                    onChange={(e) => setSelectedInfo({...selectedInfo, currency: e.target.value})}
                    className="w-full px-3 py-2 border border-slate-300 rounded text-sm focus:ring-2 focus:ring-blue-500 bg-white"
                  >
                    <option value="" disabled>—</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="USD">USD ($)</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="CHF">CHF (Fr)</option>
                    <option value="JPY">JPY (¥)</option>
                    <option value="CNY">CNY (¥)</option>
                  </select>
                </div>
              </div>
            </div>
            {bondMeta && (
              <div className="col-span-1 md:col-span-2 bg-blue-50 border border-blue-200 rounded-lg p-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                {bondMeta.issuer && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">Emittente</div>
                    <div className="font-medium text-slate-900">{bondMeta.issuer}</div>
                  </div>
                )}
                {bondMeta.coupon != null && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">Cedola</div>
                    <div className="font-medium text-slate-900">{bondMeta.coupon}%</div>
                  </div>
                )}
                {(bondMeta.maturity || bondMeta.maturity_bi) && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">Scadenza</div>
                    <div className="font-medium text-slate-900">{bondMeta.maturity_bi || bondMeta.maturity}</div>
                  </div>
                )}
                {bondMeta.ytm_gross != null && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">YTM lordo</div>
                    <div className="font-medium text-slate-900">{bondMeta.ytm_gross}%</div>
                  </div>
                )}
                {bondMeta.ytm_net != null && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">YTM netto</div>
                    <div className="font-medium text-slate-900">{bondMeta.ytm_net}%</div>
                  </div>
                )}
                {bondMeta.duration != null && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">Duration</div>
                    <div className="font-medium text-slate-900">{bondMeta.duration}</div>
                  </div>
                )}
                {bondMeta.coupon_frequency && (
                  <div>
                    <div className="text-xs text-slate-500 mb-0.5">Freq. cedola</div>
                    <div className="font-medium text-slate-900">{bondMeta.coupon_frequency}</div>
                  </div>
                )}
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Instrument</label>
              <select
                value={formData.instrument_type}
                onChange={(e) => {
                  setFormData({...formData, instrument_type: e.target.value, symbol: ''});
                  setSymbolOptions([]);
                  setSelectedInfo({ name: '', exchange: '', currency: '', ter: '' });
                  setBondMeta(null);
                  setBondLookupError(false);
                }}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="stock">Stock</option>
                <option value="etf">ETF</option>
                <option value="bond">Bond</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Type</label>
              <select
                value={formData.order_type}
                onChange={(e) => setFormData({...formData, order_type: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                {formData.instrument_type === 'bond' ? 'Nominale (€)' : 'Quantity'}
              </label>
              <input
                type="number"
                min="0.0001"
                step="any"
                value={formData.quantity}
                onChange={(e) => setFormData({...formData, quantity: e.target.value})}
                className={`w-full px-4 py-2 border ${touched.quantity && (!formData.quantity || isNaN(parseFloat(formData.quantity)) || parseFloat(formData.quantity) <= 0) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                {formData.instrument_type === 'bond' ? 'Price (% of par)' : 'Price'}
              </label>
              <div className="relative">
                {formData.instrument_type === 'bond'
                  ? <span className="absolute inset-y-0 left-3 flex items-center text-slate-500">%</span>
                  : currencySymbol && <span className="absolute inset-y-0 left-3 flex items-center text-slate-500">{currencySymbol}</span>
                }
                <input
                  type="number"
                  step="0.01"
                  value={formData.price}
                  onChange={(e) => setFormData({...formData, price: e.target.value})}
                  className={`w-full pl-8 pr-3 py-2 border ${touched.price && (!formData.price || parseFloat(formData.price) <= 0) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
                  placeholder={formData.instrument_type === 'bond' ? 'es: 96.50' : ''}
                />
              </div>
              {formData.instrument_type === 'bond' && (
                <p className="text-xs text-slate-400 mt-1">Valore = nominale × price / 100</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Commission</label>
              <div className="relative">
                {currencySymbol && <span className="absolute inset-y-0 left-3 flex items-center text-slate-500">{currencySymbol}</span>}
                <input
                  type="number"
                  step="0.01"
                  value={formData.commission}
                  onChange={(e) => setFormData({...formData, commission: e.target.value})}
                  className={`w-full ${currencySymbol ? 'pl-8 pr-3' : 'px-4'} py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500`}
                  placeholder="0"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Date</label>
              <input
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({...formData, date: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex items-end gap-3">
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {submitting && (
                  <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                  </svg>
                )}
                {submitting ? 'Saving...' : 'Submit'}
              </button>
              <button
                onClick={() => setShowForm(false)}
                className="flex-1 bg-slate-200 text-slate-700 py-2 rounded-lg hover:bg-slate-300 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="flex items-center justify-between px-4 md:px-6 pt-4">
          <div className="text-sm text-slate-600">{orders.length} orders</div>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            {showAdvanced ? 'Hide details' : 'Show more'}
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-center">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Date</th>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Symbol</th>
                {showAdvanced && <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Type</th>}
                {showAdvanced && <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Exchange</th>}
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Order</th>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Qty</th>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Price</th>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap hidden sm:table-cell">Comm.</th>
                {showAdvanced && <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">TER</th>}
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 whitespace-nowrap">Total</th>
                <th className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-700 text-right"></th>
              </tr>
            </thead>
            <tbody>
              {orders.slice(currentPage * ordersPerPage, (currentPage + 1) * ordersPerPage).map((order) => (
                <tr key={order.id} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap">{order.date}</td>
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-900 whitespace-nowrap">{order.symbol}</td>
                  {showAdvanced && <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 uppercase whitespace-nowrap">{order.instrument_type}</td>}
                  {showAdvanced && <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap">{order.exchange || '—'}</td>}
                  <td className="py-3 px-2 md:px-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                      order.order_type === 'buy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {order.order_type.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap">{order.quantity}</td>
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap">{formatCurrencyValue(order.price, order.currency)}</td>
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap hidden sm:table-cell">{formatCurrencyValue(order.commission || 0, order.currency)}</td>
                  {showAdvanced && (
                    <td className="py-3 px-2 md:px-3 text-xs md:text-sm text-slate-700 whitespace-nowrap">{formatTerValue(order.ter)}</td>
                  )}
                  <td className="py-3 px-2 md:px-3 text-xs md:text-sm font-semibold text-slate-900 whitespace-nowrap">
                    {(() => {
                      const net = order.order_type === 'buy'
                        ? order.quantity * order.price + (order.commission || 0)
                        : order.quantity * order.price - (order.commission || 0);
                      return formatCurrencyValue(net, order.currency);
                    })()}
                  </td>
                  <td className="py-3 px-2 md:px-3 text-right">
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={() => {
                          setShowForm(true);
                          setEditingOrderId(order.id);
                          setFormData({
                            symbol: order.symbol,
                            quantity: order.quantity.toString(),
                            price: order.price.toString(),
                            commission: (order.commission || 0).toString(),
                            instrument_type: order.instrument_type || 'stock',
                            order_type: order.order_type,
                            date: toISODateFromDMY(order.date) || order.date
                          });
                          setSelectedInfo({
                            name: order.name || '',
                            exchange: order.exchange || '',
                            currency: order.currency || '',
                            ter: order.ter || ''
                          });
                          setLastChosenSymbol(order.symbol);
                          setSymbolOptions([]);
                          setSearchCompleted(false);
                          scrollToForm();
                        }}
                        className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                        title="Edit order"
                      >
                        <Edit2 size={16} />
                      </button>
                      <button
                        onClick={() => setDeleteTarget(order)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        title="Delete order"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!ordersLoading && orders.length === 0 && (
          <div className="text-center py-16 text-slate-500">No orders yet</div>
        )}

        {/* Pagination Controls */}
        {orders.length > 0 && (
          <div className="px-4 py-4 border-t border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-600">Rows per page:</span>
              <select
                value={ordersPerPage}
                onChange={(e) => {
                  setOrdersPerPage(Number(e.target.value));
                  setCurrentPage(0);
                }}
                className="px-2 py-1 border border-slate-200 rounded-lg text-sm font-medium text-slate-700 bg-white hover:border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
              <span className="text-sm text-slate-600">
                {currentPage * ordersPerPage + 1}-{Math.min((currentPage + 1) * ordersPerPage, orders.length)} of {orders.length}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(0)}
                disabled={currentPage === 0}
                className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="First page"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
              <button
                onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                disabled={currentPage === 0}
                className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Previous page"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="px-3 py-1 text-sm font-medium text-slate-700">
                Page {currentPage + 1} of {Math.ceil(orders.length / ordersPerPage)}
              </span>
              <button
                onClick={() => setCurrentPage(p => Math.min(Math.ceil(orders.length / ordersPerPage) - 1, p + 1))}
                disabled={currentPage >= Math.ceil(orders.length / ordersPerPage) - 1}
                className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Next page"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <button
                onClick={() => setCurrentPage(Math.ceil(orders.length / ordersPerPage) - 1)}
                disabled={currentPage >= Math.ceil(orders.length / ordersPerPage) - 1}
                className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Last page"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>

      {deleteTarget && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50 px-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-md p-6">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Delete order?</h3>
            <p className="text-slate-600 mb-4">
              You are about to delete the order for {deleteTarget.symbol} on {deleteTarget.date}. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteTarget(null)}
                className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteOrder}
                className="px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700 transition"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default OrdersPage;
