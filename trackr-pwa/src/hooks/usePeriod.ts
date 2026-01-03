import { useState } from 'react';

type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

interface PeriodState {
  startDate: Date;
  endDate: Date;
  type: PeriodType;
}

const STORAGE_KEY = 'trackr_period';

const getInitialPeriod = (): PeriodState => {
  // Prova a caricare da localStorage
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      return {
        startDate: new Date(parsed.startDate),
        endDate: new Date(parsed.endDate),
        type: parsed.type
      };
    } catch (e) {
      // Se c'è un errore, usa il default
    }
  }

  // Default: mese corrente
  const now = new Date();
  return {
    startDate: new Date(now.getFullYear(), now.getMonth(), 1),
    endDate: new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59),
    type: 'month'
  };
};

export function usePeriod() {
  const [period, setPeriodState] = useState<PeriodState>(getInitialPeriod);

  const setPeriod = (startDate: Date, endDate: Date, type: PeriodType) => {
    const newPeriod = { startDate, endDate, type };
    setPeriodState(newPeriod);

    // Salva in localStorage
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
      type
    }));
  };

  return {
    startDate: period.startDate,
    endDate: period.endDate,
    type: period.type,
    setPeriod
  };
}
