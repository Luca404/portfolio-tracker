import { useState, useEffect } from 'react';

type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

interface PeriodSelectorProps {
  startDate: Date;
  endDate: Date;
  onPeriodChange: (startDate: Date, endDate: Date, type: PeriodType) => void;
  onCustomClick?: () => void;
  earliestDate?: Date; // Data della prima transazione per "Tutto il periodo"
}

function PeriodSelector({
  startDate,
  endDate,
  onPeriodChange,
  onCustomClick,
  earliestDate
}: PeriodSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentType, setCurrentType] = useState<PeriodType>('month');

  // Detect period type from dates when component mounts or dates change
  useEffect(() => {
    const now = new Date();
    const diffDays = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    // Check if it's a single day
    if (diffDays === 0) {
      setCurrentType('day');
    }
    // Check if it's a week (6-7 days)
    else if (diffDays >= 6 && diffDays <= 7) {
      setCurrentType('week');
    }
    // Check if it's a month (28-31 days and starts on first of month)
    else if (diffDays >= 27 && diffDays <= 31 && startDate.getDate() === 1) {
      setCurrentType('month');
    }
    // Check if it's a year (starts Jan 1, ends Dec 31)
    else if (startDate.getMonth() === 0 && startDate.getDate() === 1 &&
             endDate.getMonth() === 11 && endDate.getDate() === 31) {
      setCurrentType('year');
    }
    // Check if it's a very long period (could be "all")
    else if (diffDays > 365) {
      setCurrentType('all');
    }
    // Otherwise it's custom
    else {
      setCurrentType('custom');
    }
  }, [startDate, endDate]);

  const formatDateRange = (start: Date, end: Date, type: PeriodType) => {
    // Per giorno mostra solo la data singola
    if (type === 'day') {
      return start.toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: 'numeric' });
    }

    const startStr = start.toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: 'numeric' });
    const endStr = end.toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: 'numeric' });
    return `${startStr} - ${endStr}`;
  };

  const getPeriodLabel = () => {
    const monthNames = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                        'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre'];

    if (currentType === 'month') {
      return `${monthNames[startDate.getMonth()]} ${startDate.getFullYear()}`;
    } else if (currentType === 'year') {
      return `${startDate.getFullYear()}`;
    } else if (currentType === 'week') {
      return 'Settimana';
    } else if (currentType === 'day') {
      return 'Giorno';
    } else if (currentType === 'all') {
      return 'Tutto il periodo';
    } else {
      return 'Periodo Personalizzato';
    }
  };

  const handlePeriodSelect = (type: PeriodType) => {
    const now = new Date();
    let start: Date;
    let end: Date;

    switch (type) {
      case 'day':
        start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);
        break;

      case 'week':
        const dayOfWeek = now.getDay();
        const diff = dayOfWeek === 0 ? 6 : dayOfWeek - 1; // Lunedì = inizio settimana
        start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - diff);
        end = new Date(start.getFullYear(), start.getMonth(), start.getDate() + 6, 23, 59, 59);
        break;

      case 'month':
        start = new Date(now.getFullYear(), now.getMonth(), 1);
        end = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59);
        break;

      case 'year':
        start = new Date(now.getFullYear(), 0, 1);
        end = new Date(now.getFullYear(), 11, 31, 23, 59, 59);
        break;

      case 'all':
        // Usa earliestDate se disponibile, altrimenti usa un anno fa
        start = earliestDate || new Date(now.getFullYear() - 1, 0, 1);
        end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);
        break;

      case 'custom':
        if (onCustomClick) {
          onCustomClick();
        }
        setIsOpen(false);
        return;

      default:
        start = new Date(now.getFullYear(), now.getMonth(), 1);
        end = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59);
    }

    setCurrentType(type);
    onPeriodChange(start, end, type);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full bg-white dark:bg-gray-800 rounded-lg border border-gray-300 dark:border-gray-600 p-4 text-center"
      >
        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
          {getPeriodLabel()}
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          {formatDateRange(startDate, endDate, currentType)}
        </div>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-50 mt-2 w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-300 dark:border-gray-600 p-3">
            <div className="space-y-2">
              <button
                type="button"
                onClick={() => handlePeriodSelect('day')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'day'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Giorno</div>
                <div className="text-sm opacity-75">Oggi</div>
              </button>

              <button
                type="button"
                onClick={() => handlePeriodSelect('week')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'week'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Settimana</div>
                <div className="text-sm opacity-75">Settimana corrente</div>
              </button>

              <button
                type="button"
                onClick={() => handlePeriodSelect('month')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'month'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Mese</div>
                <div className="text-sm opacity-75">Mese corrente</div>
              </button>

              <button
                type="button"
                onClick={() => handlePeriodSelect('year')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'year'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Anno</div>
                <div className="text-sm opacity-75">Anno corrente</div>
              </button>

              <button
                type="button"
                onClick={() => handlePeriodSelect('all')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'all'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Tutto il Periodo</div>
                <div className="text-sm opacity-75">Tutte le transazioni</div>
              </button>

              <button
                type="button"
                onClick={() => handlePeriodSelect('custom')}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  currentType === 'custom'
                    ? 'bg-primary-500 text-white'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100'
                }`}
              >
                <div className="font-medium">Personalizzato</div>
                <div className="text-sm opacity-75">Scegli le date</div>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default PeriodSelector;
export type { PeriodType };
