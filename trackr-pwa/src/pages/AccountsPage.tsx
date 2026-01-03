import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import LoadingSpinner from '../components/common/LoadingSpinner';
import type { Account, AccountFormData } from '../types';

const ACCOUNT_ICONS = ['💳', '🏦', '💰', '💵', '💶', '💷', '💴', '🪙', '💸', '🏧', '📱', '💎'];

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedAccount, setSelectedAccount] = useState<Account | null>(null);
  const [formData, setFormData] = useState<AccountFormData>({
    name: '',
    icon: '💳',
    initial_balance: 0,
  });

  useEffect(() => {
    loadAccounts();
  }, []);

  const loadAccounts = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getAccounts();
      const accountsArray = Array.isArray(data) ? data : [];
      setAccounts(accountsArray);

      // Se non ci sono conti, crea quelli di default
      if (accountsArray.length === 0) {
        await createDefaultAccounts();
      }
    } catch (error) {
      console.error('Errore caricamento conti:', error);
      setAccounts([]);
    } finally {
      setIsLoading(false);
    }
  };

  const createDefaultAccounts = async () => {
    try {
      // Crea Conto Corrente
      await apiService.createAccount({
        name: 'Conto Corrente',
        icon: '🏦',
        initial_balance: 0,
      });

      // Crea Contanti
      await apiService.createAccount({
        name: 'Contanti',
        icon: '💵',
        initial_balance: 0,
      });

      // Ricarica i conti
      const data = await apiService.getAccounts();
      setAccounts(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Errore creazione conti di default:', error);
    }
  };

  const handleOpenModal = (account?: Account) => {
    if (account) {
      setIsEditMode(true);
      setSelectedAccount(account);
      setFormData({
        name: account.name,
        icon: account.icon,
        initial_balance: account.initial_balance,
      });
    } else {
      setIsEditMode(false);
      setSelectedAccount(null);
      setFormData({
        name: '',
        icon: '💳',
        initial_balance: 0,
      });
    }
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedAccount(null);
    setIsEditMode(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      if (isEditMode && selectedAccount) {
        await apiService.updateAccount(selectedAccount.id, formData);
      } else {
        await apiService.createAccount(formData);
      }
      await loadAccounts();
      handleCloseModal();
    } catch (error) {
      console.error('Errore salvataggio conto:', error);
      alert('Errore durante il salvataggio del conto');
    }
  };

  const handleDelete = async (id: number) => {
    if (confirm('Sei sicuro di voler eliminare questo conto?')) {
      try {
        await apiService.deleteAccount(id);
        await loadAccounts();
      } catch (error) {
        console.error('Errore eliminazione conto:', error);
        alert('Errore durante l\'eliminazione del conto');
      }
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Conti
        </h1>

        {/* Lista conti */}
        {accounts.length === 0 ? (
          <div className="card text-center py-12">
            <div className="text-6xl mb-4">🏦</div>
            <div className="text-gray-500 dark:text-gray-400 mb-2">
              Nessun conto trovato
            </div>
            <div className="text-sm text-gray-400 dark:text-gray-500">
              Aggiungi il tuo primo conto per iniziare
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {accounts.map((account) => (
              <div
                key={account.id}
                className="card hover:shadow-lg transition-shadow cursor-pointer"
                onClick={() => handleOpenModal(account)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="text-4xl">{account.icon}</div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                        {account.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Saldo iniziale
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${
                      account.initial_balance >= 0
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}>
                      {formatCurrency(account.initial_balance)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* FAB per aggiungere conto */}
        <FAB onClick={() => handleOpenModal()} />

        {/* Modal per creare/modificare conto */}
        <Modal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          title={isEditMode ? 'Modifica Conto' : 'Nuovo Conto'}
        >
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Nome */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Nome
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="es. Conto Principale"
                required
              />
            </div>

            {/* Icona */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Icona
              </label>
              <div className="grid grid-cols-6 gap-2">
                {ACCOUNT_ICONS.map((icon) => (
                  <button
                    key={icon}
                    type="button"
                    onClick={() => setFormData({ ...formData, icon })}
                    className={`p-3 text-2xl rounded-lg border-2 transition-colors ${
                      formData.icon === icon
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                    }`}
                  >
                    {icon}
                  </button>
                ))}
              </div>
            </div>

            {/* Saldo iniziale */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Saldo Iniziale
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.initial_balance}
                onChange={(e) => setFormData({ ...formData, initial_balance: parseFloat(e.target.value) || 0 })}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="0.00"
                required
              />
            </div>

            {/* Azioni */}
            <div className="flex gap-3 pt-4">
              {isEditMode && (
                <button
                  type="button"
                  onClick={() => {
                    if (selectedAccount) {
                      handleDelete(selectedAccount.id);
                      handleCloseModal();
                    }
                  }}
                  className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors"
                >
                  Elimina
                </button>
              )}
              <button
                type="button"
                onClick={handleCloseModal}
                className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                Annulla
              </button>
              <button
                type="submit"
                className="flex-1 px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors"
              >
                {isEditMode ? 'Salva' : 'Crea'}
              </button>
            </div>
          </form>
        </Modal>
      </div>
    </Layout>
  );
}
