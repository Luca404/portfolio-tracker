import { useState } from 'react';
import { apiService } from '../services/api';

export default function SettingsPage() {
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleExport = async () => {
    try {
      await apiService.exportData();
      setMessage({ type: 'success', text: 'Backup esportato con successo!' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Errore durante l\'esportazione' });
      console.error('Export error:', error);
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setImporting(true);
    setMessage(null);

    try {
      await apiService.importData(file);
      setMessage({ type: 'success', text: 'Dati importati con successo! Ricarica la pagina.' });

      // Ricarica la pagina dopo 2 secondi
      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } catch (error) {
      setMessage({ type: 'error', text: 'Errore durante l\'importazione' });
      console.error('Import error:', error);
    } finally {
      setImporting(false);
    }
  };

  const user = apiService.getCurrentUser();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20">
      <div className="max-w-2xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">Impostazioni</h1>

        {/* Info utente */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Profilo</h2>
          <div className="space-y-2">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-medium">Utente:</span> {user?.name || 'Utente Locale'}
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-medium">ID:</span> {user?.id || 'N/A'}
            </p>
          </div>
        </div>

        {/* Modalità app */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Modalità App</h2>
          <div className="flex items-center justify-between p-4 bg-sky-50 dark:bg-sky-900/20 rounded-lg border border-sky-200 dark:border-sky-800">
            <div>
              <p className="font-medium text-sky-900 dark:text-sky-100">Modalità Offline</p>
              <p className="text-sm text-sky-700 dark:text-sky-300">
                I dati sono salvati localmente sul dispositivo
              </p>
            </div>
            <svg className="w-8 h-8 text-sky-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        </div>

        {/* Backup e Ripristino */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Backup e Ripristino</h2>

          <div className="space-y-4">
            {/* Export */}
            <div>
              <button
                onClick={handleExport}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-sky-600 hover:bg-sky-700 text-white rounded-lg font-medium transition-colors"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                </svg>
                Esporta Backup
              </button>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Scarica un file JSON con tutti i tuoi dati
              </p>
            </div>

            {/* Import */}
            <div>
              <label className="block">
                <input
                  type="file"
                  accept=".json"
                  onChange={handleImport}
                  disabled={importing}
                  className="hidden"
                  id="import-file"
                />
                <label
                  htmlFor="import-file"
                  className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors cursor-pointer ${
                    importing
                      ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  {importing ? 'Importazione...' : 'Importa Backup'}
                </label>
              </label>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Carica un file JSON precedentemente esportato
              </p>
              <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                ⚠️ Attenzione: sostituirà tutti i dati esistenti
              </p>
            </div>
          </div>

          {/* Messaggi */}
          {message && (
            <div
              className={`mt-4 p-4 rounded-lg ${
                message.type === 'success'
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-200 border border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-800'
              }`}
            >
              {message.text}
            </div>
          )}
        </div>

        {/* Info PWA */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Installazione App</h2>
          <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
            <p>
              Questa è una Progressive Web App (PWA). Puoi installarla sul tuo dispositivo:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>
                <span className="font-medium">Android:</span> Tocca i 3 puntini in alto → "Aggiungi a schermata Home"
              </li>
              <li>
                <span className="font-medium">iOS:</span> Tocca il pulsante "Condividi" → "Aggiungi a Home"
              </li>
              <li>
                <span className="font-medium">Desktop:</span> Clicca sull'icona di installazione nella barra degli indirizzi
              </li>
            </ul>
            <p className="text-xs pt-2 border-t border-gray-200 dark:border-gray-700">
              Una volta installata, funzionerà come un'app nativa, anche offline!
            </p>
          </div>
        </div>

        {/* Logout */}
        <div className="mt-8">
          <button
            onClick={() => apiService.logout()}
            className="w-full px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
          >
            Esci
          </button>
        </div>
      </div>
    </div>
  );
}
