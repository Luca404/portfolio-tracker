import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import LoadingSpinner from '../components/common/LoadingSpinner';
import type { Portfolio, PortfolioFormData } from '../types';

const CACHE_KEY = 'portfolios_cache';
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 1 giorno in millisecondi

interface CachedData {
  portfolios: Portfolio[];
  timestamp: number;
}

export default function PortfoliosPage() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);

  useEffect(() => {
    loadPortfolios();
  }, []);

  const loadPortfolios = async (forceRefresh = false) => {
    setIsLoading(true);
    try {
      // Controlla la cache se non è un refresh forzato
      if (!forceRefresh) {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
          const cachedData: CachedData = JSON.parse(cached);
          const now = Date.now();
          // Se la cache è ancora valida (meno di 1 giorno)
          if (now - cachedData.timestamp < CACHE_DURATION) {
            setPortfolios(cachedData.portfolios);
            setIsLoading(false);
            return;
          }
        }
      }

      // Carica da API
      const data = await apiService.getPortfolios();
      setPortfolios(data);

      // Salva in cache
      const cacheData: CachedData = {
        portfolios: data,
        timestamp: Date.now(),
      };
      localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
    } catch (error) {
      console.error('Errore caricamento portafogli:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreatePortfolio = () => {
    setSelectedPortfolio(null);
    setIsEditMode(false);
    setIsModalOpen(true);
  };

  const handleEditPortfolio = (portfolio: Portfolio) => {
    setSelectedPortfolio(portfolio);
    setIsEditMode(true);
    setIsModalOpen(true);
  };

  const handleSubmit = async (data: PortfolioFormData) => {
    try {
      if (isEditMode && selectedPortfolio) {
        await apiService.updatePortfolio(selectedPortfolio.id, data);
      } else {
        await apiService.createPortfolio(data);
      }
      await loadPortfolios(true); // Force refresh dopo creazione/modifica
      setIsModalOpen(false);
    } catch (error) {
      console.error('Errore salvataggio portafoglio:', error);
    }
  };

  const handleDelete = async (id: number) => {
    if (confirm('Sei sicuro di voler eliminare questo portafoglio?')) {
      try {
        await apiService.deletePortfolio(id);
        await loadPortfolios(true); // Force refresh dopo eliminazione
      } catch (error) {
        console.error('Errore eliminazione portafoglio:', error);
      }
    }
  };

  const formatCurrency = (value: number, currency: string = 'EUR') => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-6">
        {/* Lista portafogli */}
        {portfolios.length === 0 ? (
          <div className="card text-center py-12">
            <div className="text-6xl mb-4">📊</div>
            <div className="text-gray-500 dark:text-gray-400 mb-2">
              Nessun portafoglio trovato
            </div>
            <div className="text-sm text-gray-400 dark:text-gray-500">
              Crea il tuo primo portafoglio di investimenti
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {portfolios.map((portfolio) => (
              <div
                key={portfolio.id}
                className="card cursor-pointer hover:shadow-lg transition-shadow"
                onClick={() => handleEditPortfolio(portfolio)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {portfolio.name}
                      </span>
                    </div>
                    {portfolio.description && (
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {portfolio.description}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(portfolio.id);
                    }}
                    className="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 p-2"
                  >
                    🗑️
                  </button>
                </div>

                {/* Recap con valore e P/L */}
                {portfolio.total_value !== undefined && (
                  <div className="grid grid-cols-2 gap-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Valore Attuale
                      </div>
                      <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                        {formatCurrency(portfolio.total_value, portfolio.reference_currency)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        P/L
                      </div>
                      <div className={`text-lg font-semibold ${
                        (portfolio.total_gain_loss || 0) >= 0
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {formatCurrency(portfolio.total_gain_loss || 0, portfolio.reference_currency)}
                        {portfolio.total_gain_loss_pct !== undefined && (
                          <span className="text-sm ml-1">
                            ({portfolio.total_gain_loss_pct >= 0 ? '+' : ''}{portfolio.total_gain_loss_pct.toFixed(2)}%)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* FAB per creare nuovo portafoglio */}
        <FAB onClick={handleCreatePortfolio} />

        {/* Modal per creare/modificare portafoglio */}
        <Modal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          title={isEditMode ? 'Modifica Portafoglio' : 'Nuovo Portafoglio'}
        >
          <PortfolioForm
            onSubmit={handleSubmit}
            onCancel={() => setIsModalOpen(false)}
            initialData={selectedPortfolio ? {
              name: selectedPortfolio.name,
              description: selectedPortfolio.description,
              initial_capital: selectedPortfolio.initial_capital,
              reference_currency: selectedPortfolio.reference_currency,
              risk_free_source: selectedPortfolio.risk_free_source,
              market_benchmark: selectedPortfolio.market_benchmark,
            } : undefined}
            isEditMode={isEditMode}
          />
        </Modal>
      </div>
    </Layout>
  );
}

// Form per creare/modificare portafoglio
interface PortfolioFormProps {
  onSubmit: (data: PortfolioFormData) => Promise<void>;
  onCancel: () => void;
  initialData?: PortfolioFormData;
  isEditMode?: boolean;
}

function PortfolioForm({ onSubmit, onCancel, initialData, isEditMode }: PortfolioFormProps) {
  const [name, setName] = useState(initialData?.name || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [initialCapital, setInitialCapital] = useState(initialData?.initial_capital?.toString() || '10000');
  const [currency, setCurrency] = useState(initialData?.reference_currency || 'EUR');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit({
      name,
      description,
      initial_capital: parseFloat(initialCapital),
      reference_currency: currency,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Nome Portafoglio
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="input"
          placeholder="Es: Il mio portafoglio"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Descrizione (opzionale)
        </label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="input"
          placeholder="Descrizione del portafoglio..."
          rows={3}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Capitale Iniziale
        </label>
        <input
          type="number"
          step="0.01"
          value={initialCapital}
          onChange={(e) => setInitialCapital(e.target.value)}
          className="input"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Valuta
        </label>
        <select
          value={currency}
          onChange={(e) => setCurrency(e.target.value)}
          className="input"
        >
          <option value="EUR">EUR (€)</option>
          <option value="USD">USD ($)</option>
          <option value="GBP">GBP (£)</option>
        </select>
      </div>

      <div className="flex gap-3 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 btn-secondary"
        >
          Annulla
        </button>
        <button
          type="submit"
          className="flex-1 btn-primary"
        >
          {isEditMode ? 'Salva' : 'Crea'}
        </button>
      </div>
    </form>
  );
}
