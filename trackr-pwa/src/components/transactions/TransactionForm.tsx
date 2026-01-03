import { useState, useEffect, type FormEvent } from 'react';
import type { TransactionFormData, TransactionType, Category, Subcategory, Account } from '../../types';
import { apiService } from '../../services/api';

interface TransactionFormProps {
  onSubmit: (data: TransactionFormData) => Promise<void>;
  onCancel: () => void;
  initialData?: TransactionFormData;
  isEditMode?: boolean;
  onDelete?: () => Promise<void>;
}

export default function TransactionForm({ onSubmit, onCancel, initialData, isEditMode, onDelete }: TransactionFormProps) {
  const [currentType, setCurrentType] = useState<TransactionType>(initialData?.type || 'expense');
  const [categories, setCategories] = useState<Category[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
  const [selectedSubcategory, setSelectedSubcategory] = useState<Subcategory | null>(null);
  const [selectedAccount, setSelectedAccount] = useState<Account | null>(null);
  const [amount, setAmount] = useState<string>(initialData?.amount.toString() || '');
  const [date, setDate] = useState<string>(initialData?.date || new Date().toISOString().split('T')[0]);
  const [description, setDescription] = useState<string>(initialData?.description || '');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showDateSelector, setShowDateSelector] = useState(false);
  const [showCategoryPicker, setShowCategoryPicker] = useState(false);
  const [showAccountPicker, setShowAccountPicker] = useState(false);
  const [showCurrencyPicker, setShowCurrencyPicker] = useState(false);
  const [currency, setCurrency] = useState<string>('EUR');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Carica categorie e conti all'avvio
  useEffect(() => {
    loadCategories();
    loadAccounts();
  }, []);

  // Carica categorie quando cambia il tipo
  useEffect(() => {
    loadCategories();
    setSelectedCategory(null);
    setSelectedSubcategory(null);
  }, [currentType]);

  useEffect(() => {
    if (isEditMode && initialData && categories.length > 0) {
      const category = categories.find(c => c.name === initialData.category);
      if (category) {
        setSelectedCategory(category);
        if (initialData.subcategory) {
          const subcategory = category.subcategories?.find(s => s.name === initialData.subcategory);
          if (subcategory) {
            setSelectedSubcategory(subcategory);
          }
        }
      }
    }
  }, [isEditMode, initialData, categories]);

  const loadCategories = async () => {
    try {
      const data = await apiService.getCategories();
      setCategories(data.filter(c => !c.category_type || c.category_type === currentType));
    } catch (err) {
      console.error('Errore caricamento categorie:', err);
    }
  };

  const loadAccounts = async () => {
    try {
      const data = await apiService.getAccounts();
      setAccounts(data);
      if (data.length > 0 && !selectedAccount) {
        setSelectedAccount(data[0]);
      }
    } catch (err) {
      console.error('Errore caricamento conti:', err);
    }
  };

  const handleNumberClick = (num: string) => {
    if (num === '.' && amount.includes('.')) return;
    setAmount(prev => prev === '0' ? num : prev + num);
  };

  const handleBackspace = () => {
    setAmount(prev => prev.length > 1 ? prev.slice(0, -1) : '0');
  };

  const getCurrencySymbol = (curr: string) => {
    const symbols: Record<string, string> = {
      'EUR': '€',
      'USD': '$',
      'GBP': '£',
      'JPY': '¥',
      'CHF': 'Fr',
    };
    return symbols[curr] || curr;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!selectedCategory) {
      setError('Seleziona una categoria');
      return;
    }

    if (!selectedAccount) {
      setError('Seleziona un conto');
      return;
    }

    const amountNum = parseFloat(amount) || 0;
    if (amountNum <= 0) {
      setError('Inserisci un importo valido');
      return;
    }

    setError('');
    setIsLoading(true);

    try {
      await onSubmit({
        type: currentType,
        category: selectedCategory.name,
        subcategory: selectedSubcategory?.name,
        amount: amountNum,
        description,
        date,
      });
      onCancel();
    } catch (err: any) {
      setError(err.response?.data?.message || 'Errore durante il salvataggio');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString('it-IT', {
      weekday: 'short',
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  };

  const typeButtons = [
    { type: 'expense' as TransactionType, label: 'Uscita', icon: '💸', color: 'red' },
    { type: 'income' as TransactionType, label: 'Entrata', icon: '💰', color: 'green' },
    { type: 'investment' as TransactionType, label: 'Investimento', icon: '📈', color: 'blue' },
    { type: 'transfer' as TransactionType, label: 'Trasferimento', icon: '🔄', color: 'purple' },
  ];

  const handleDateQuickSelect = (option: 'today' | 'yesterday') => {
    const today = new Date();
    if (option === 'today') {
      setDate(today.toISOString().split('T')[0]);
    } else {
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      setDate(yesterday.toISOString().split('T')[0]);
    }
    setShowDateSelector(false);
    setShowDatePicker(false);
  };

  // Se non è stata selezionata una categoria, mostra la griglia
  if (!selectedCategory) {
    return (
      <div className="space-y-4">
        {/* Tab tipi transazione */}
        <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
          {typeButtons.map(({ type, label, icon }) => (
            <button
              key={type}
              type="button"
              onClick={() => setCurrentType(type)}
              className={`flex-1 px-3 py-2 font-medium text-sm transition-colors flex flex-col items-center ${
                currentType === type
                  ? 'border-b-2 border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              <span className="text-xl mb-0.5">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Griglia categorie */}
        <div>
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Seleziona categoria
          </div>
          <div className="grid grid-cols-3 gap-3 max-h-[60vh] overflow-y-auto">
            {categories.map((category) => (
              <button
                key={category.id}
                type="button"
                onClick={() => setSelectedCategory(category)}
                className="flex flex-col items-center justify-center p-4 rounded-xl border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-colors bg-white dark:bg-gray-800"
              >
                <span className="text-4xl mb-2">{category.icon}</span>
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300 text-center">
                  {category.name}
                </span>
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={onCancel}
          className="w-full btn-secondary"
        >
          Annulla
        </button>
      </div>
    );
  }

  // Form con tastierino numerico
  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Categoria e Conto selezionati */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setShowCategoryPicker(true)}
          className="flex-1 flex items-center gap-2 p-3 rounded-lg border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 hover:border-primary-500 transition-colors"
        >
          <span className="text-2xl">{selectedCategory.icon}</span>
          <div className="flex-1 text-left">
            <div className="text-xs text-gray-500 dark:text-gray-400">Categoria</div>
            <div className="font-medium text-gray-900 dark:text-gray-100">{selectedCategory.name}</div>
          </div>
          <span className="text-gray-400">›</span>
        </button>

        <button
          type="button"
          onClick={() => setShowAccountPicker(true)}
          className="flex-1 flex items-center gap-2 p-3 rounded-lg border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 hover:border-primary-500 transition-colors"
        >
          <span className="text-2xl">{selectedAccount?.icon || '💳'}</span>
          <div className="flex-1 text-left">
            <div className="text-xs text-gray-500 dark:text-gray-400">Conto</div>
            <div className="font-medium text-gray-900 dark:text-gray-100">{selectedAccount?.name || 'Seleziona'}</div>
          </div>
          <span className="text-gray-400">›</span>
        </button>
      </div>

      {/* Sottocategorie */}
      {selectedCategory.subcategories && selectedCategory.subcategories.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Sottocategoria (opzionale)</div>
          <div className="flex gap-2 flex-wrap">
            <button
              type="button"
              onClick={() => setSelectedSubcategory(null)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                !selectedSubcategory
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              Nessuna
            </button>
            {selectedCategory.subcategories.map((sub) => (
              <button
                key={sub.id}
                type="button"
                onClick={() => setSelectedSubcategory(sub)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors flex items-center gap-1 ${
                  selectedSubcategory?.id === sub.id
                    ? 'bg-primary-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <span>{sub.icon}</span>
                <span>{sub.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Display importo */}
      <div className="text-center py-4">
        <div className="text-5xl font-bold text-gray-900 dark:text-gray-100">
          {getCurrencySymbol(currency)} {amount || '0'}
        </div>
      </div>

      {/* Tastierino numerico con colonna comandi a destra */}
      <div className="flex gap-2">
        {/* Griglia numeri 3x4 (sinistra) */}
        <div className="flex-1 grid grid-cols-3 gap-2">
          {/* Riga 1: 1 2 3 */}
          <button type="button" onClick={() => handleNumberClick('1')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">1</button>
          <button type="button" onClick={() => handleNumberClick('2')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">2</button>
          <button type="button" onClick={() => handleNumberClick('3')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">3</button>

          {/* Riga 2: 4 5 6 */}
          <button type="button" onClick={() => handleNumberClick('4')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">4</button>
          <button type="button" onClick={() => handleNumberClick('5')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">5</button>
          <button type="button" onClick={() => handleNumberClick('6')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">6</button>

          {/* Riga 3: 7 8 9 */}
          <button type="button" onClick={() => handleNumberClick('7')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">7</button>
          <button type="button" onClick={() => handleNumberClick('8')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">8</button>
          <button type="button" onClick={() => handleNumberClick('9')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">9</button>

          {/* Riga 4: € 0 . */}
          <button type="button" onClick={() => setShowCurrencyPicker(true)} className="h-14 text-lg font-semibold rounded-lg bg-primary-100 dark:bg-primary-900/30 hover:bg-primary-200 dark:hover:bg-primary-900/50 text-primary-600 dark:text-primary-400 transition-colors">{getCurrencySymbol(currency)}</button>
          <button type="button" onClick={() => handleNumberClick('0')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">0</button>
          <button type="button" onClick={() => handleNumberClick('.')} className="h-14 text-2xl font-semibold rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">.</button>
        </div>

        {/* Colonna comandi (destra) */}
        <div className="flex flex-col gap-2">
          {/* Backspace */}
          <button type="button" onClick={handleBackspace} className="h-14 w-14 text-2xl rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 transition-colors">←</button>

          {/* OK (doppia altezza - prende lo spazio rimanente) */}
          <button type="submit" disabled={isLoading || !selectedCategory || !selectedAccount || parseFloat(amount) <= 0} className="flex-1 w-14 rounded-lg bg-green-500 hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold text-3xl transition-colors flex items-center justify-center">{isLoading ? '...' : '✓'}</button>

          {/* Calendario */}
          <button type="button" onClick={() => setShowDateSelector(true)} className="h-14 w-14 text-2xl rounded-lg bg-primary-100 dark:bg-primary-900/30 hover:bg-primary-200 dark:hover:bg-primary-900/50 text-primary-600 dark:text-primary-400 transition-colors">📅</button>
        </div>
      </div>

      {/* Data selezionata */}
      <div className="text-center text-sm text-gray-600 dark:text-gray-400">
        {formatDate(date)}
      </div>

      {/* Descrizione (opzionale) */}
      <div>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Descrizione (opzionale)"
          className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
        />
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Pulsante elimina solo in edit mode */}
      {isEditMode && onDelete && (
        <button
          type="button"
          onClick={async () => {
            if (confirm('Sei sicuro di voler eliminare questa transazione?')) {
              setIsLoading(true);
              try {
                await onDelete();
              } catch (err: any) {
                setError(err.response?.data?.message || 'Errore durante l\'eliminazione');
                setIsLoading(false);
              }
            }
          }}
          className="w-full px-4 py-3 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors font-medium"
          disabled={isLoading}
        >
          🗑️ Elimina
        </button>
      )}

      {/* Modal cambio categoria */}
      {showCategoryPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => setShowCategoryPicker(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Cambia Categoria</h3>
            <div className="grid grid-cols-3 gap-3">
              {categories.map((category) => (
                <button
                  key={category.id}
                  type="button"
                  onClick={() => {
                    setSelectedCategory(category);
                    setSelectedSubcategory(null);
                    setShowCategoryPicker(false);
                  }}
                  className="flex flex-col items-center p-3 rounded-lg border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
                >
                  <span className="text-3xl mb-1">{category.icon}</span>
                  <span className="text-xs text-center">{category.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Modal cambio conto */}
      {showAccountPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => setShowAccountPicker(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Seleziona Conto</h3>
            <div className="space-y-2">
              {accounts.map((account) => (
                <button
                  key={account.id}
                  type="button"
                  onClick={() => {
                    setSelectedAccount(account);
                    setShowAccountPicker(false);
                  }}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg border-2 transition-colors ${
                    selectedAccount?.id === account.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary-500'
                  }`}
                >
                  <span className="text-2xl">{account.icon}</span>
                  <span className="font-medium">{account.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Modal selettore valuta */}
      {showCurrencyPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => setShowCurrencyPicker(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-sm w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Seleziona Valuta</h3>
            <div className="space-y-2">
              {[
                { code: 'EUR', name: 'Euro', symbol: '€' },
                { code: 'USD', name: 'Dollaro USA', symbol: '$' },
                { code: 'GBP', name: 'Sterlina Britannica', symbol: '£' },
                { code: 'JPY', name: 'Yen Giapponese', symbol: '¥' },
                { code: 'CHF', name: 'Franco Svizzero', symbol: 'Fr' },
              ].map((curr) => (
                <button
                  key={curr.code}
                  type="button"
                  onClick={() => {
                    setCurrency(curr.code);
                    setShowCurrencyPicker(false);
                  }}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg border-2 transition-colors ${
                    currency === curr.code
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary-500'
                  }`}
                >
                  <span className="text-2xl font-bold w-12">{curr.symbol}</span>
                  <div className="flex-1 text-left">
                    <div className="font-medium text-gray-900 dark:text-gray-100">{curr.name}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{curr.code}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Modal selettore data */}
      {showDateSelector && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => {
          setShowDateSelector(false);
          setShowDatePicker(false);
        }}>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-sm w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Seleziona Data</h3>

            <div className="space-y-2">
              {/* Oggi */}
              <button
                type="button"
                onClick={() => handleDateQuickSelect('today')}
                className="w-full flex items-center gap-3 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
              >
                <span className="text-2xl">📅</span>
                <div className="flex-1 text-left">
                  <div className="font-medium text-gray-900 dark:text-gray-100">Oggi</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {new Date().toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: 'numeric' })}
                  </div>
                </div>
              </button>

              {/* Ieri */}
              <button
                type="button"
                onClick={() => handleDateQuickSelect('yesterday')}
                className="w-full flex items-center gap-3 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
              >
                <span className="text-2xl">⏮️</span>
                <div className="flex-1 text-left">
                  <div className="font-medium text-gray-900 dark:text-gray-100">Ieri</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(() => {
                      const yesterday = new Date();
                      yesterday.setDate(yesterday.getDate() - 1);
                      return yesterday.toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: 'numeric' });
                    })()}
                  </div>
                </div>
              </button>

              {/* Seleziona giorno */}
              <button
                type="button"
                onClick={() => setShowDatePicker(true)}
                className="w-full flex items-center gap-3 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
              >
                <span className="text-2xl">🗓️</span>
                <div className="flex-1 text-left">
                  <div className="font-medium text-gray-900 dark:text-gray-100">Seleziona giorno</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Scegli una data specifica</div>
                </div>
              </button>

              {/* Calendario (appare solo se "Seleziona giorno" è cliccato) */}
              {showDatePicker && (
                <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => {
                      setDate(e.target.value);
                      setShowDatePicker(false);
                      setShowDateSelector(false);
                    }}
                    className="w-full px-4 py-3 rounded-lg border-2 border-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-lg"
                    autoFocus
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </form>
  );
}
