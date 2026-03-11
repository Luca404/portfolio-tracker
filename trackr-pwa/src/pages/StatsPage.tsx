import { useState, useMemo } from 'react';
import { useData } from '../contexts/DataContext';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PeriodSelector from '../components/common/PeriodSelector';
import DateRangePicker from '../components/common/DateRangePicker';
import { usePeriod } from '../hooks/usePeriod';
import type { Transaction } from '../types';

type StatsFilter = 'expense' | 'income' | 'investment';
type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

interface CategoryStat {
  name: string;
  icon: string;
  amount: number;
  percentage: number;
  count: number;
}

export default function StatsPage() {
  const { transactions: allTransactions, categories, isLoading } = useData();
  const [filter, setFilter] = useState<StatsFilter>('expense');
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);

  // Period state - condiviso tra le pagine
  const { startDate, endDate, type: periodType, setPeriod } = usePeriod();

  // Filtra le transazioni in base al periodo selezionato
  const transactions = useMemo(() => {
    return allTransactions.filter(transaction => {
      const transactionDate = new Date(transaction.date);
      return transactionDate >= startDate && transactionDate <= endDate;
    });
  }, [allTransactions, startDate, endDate]);

  const handlePeriodChange = (start: Date, end: Date, type: PeriodType) => {
    setPeriod(start, end, type);
  };

  const handleCustomPeriodConfirm = (start: Date, end: Date) => {
    setPeriod(start, end, 'custom');
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  // Calcola totali del periodo
  const totalIncome = transactions
    .filter(t => t.type === 'income')
    .reduce((sum, t) => sum + t.amount, 0);

  const totalExpense = transactions
    .filter(t => t.type === 'expense')
    .reduce((sum, t) => sum + t.amount, 0);

  const totalInvestment = transactions
    .filter(t => t.type === 'investment')
    .reduce((sum, t) => sum + t.amount, 0);

  // Saldo del periodo (entrate - uscite, escludendo investimenti e trasferimenti)
  const periodBalance = totalIncome - totalExpense;

  // Calcola statistiche per categoria
  const filteredTransactions = transactions.filter(t => t.type === filter);
  const total = filteredTransactions.reduce((sum, t) => sum + Math.abs(t.amount), 0);

  console.log('StatsPage render:', {
    periodType,
    transactionsCount: transactions.length,
    filteredCount: filteredTransactions.length,
    startDate: startDate.toISOString(),
    endDate: endDate.toISOString()
  });

  const categoryStats: CategoryStat[] = filteredTransactions.reduce((acc, t) => {
    const existing = acc.find(c => c.name === t.category);
    if (existing) {
      existing.amount += Math.abs(t.amount);
      existing.count += 1;
    } else {
      const category = categories.find(c => c.name === t.category);
      acc.push({
        name: t.category,
        icon: category?.icon || '📌',
        amount: Math.abs(t.amount),
        percentage: 0,
        count: 1,
      });
    }
    return acc;
  }, [] as CategoryStat[]);

  // Calcola percentuali
  categoryStats.forEach(stat => {
    stat.percentage = total > 0 ? (stat.amount / total) * 100 : 0;
  });

  // Ordina per amount decrescente
  categoryStats.sort((a, b) => b.amount - a.amount);

  // Genera i periodi da mostrare nel grafico
  const generatePeriods = () => {
    const periods: { label: string; date: Date }[] = [];
    const current = new Date(startDate);

    if (periodType === 'day') {
      // Mostra ogni giorno
      while (current <= endDate) {
        periods.push({
          label: current.getDate().toString(),
          date: new Date(current)
        });
        current.setDate(current.getDate() + 1);
      }
    } else if (periodType === 'week') {
      // Mostra ogni giorno della settimana
      while (current <= endDate) {
        periods.push({
          label: current.toLocaleDateString('it-IT', { weekday: 'short' }),
          date: new Date(current)
        });
        current.setDate(current.getDate() + 1);
      }
    } else if (periodType === 'month') {
      // Mostra ogni giorno del mese
      while (current <= endDate) {
        periods.push({
          label: current.getDate().toString(),
          date: new Date(current)
        });
        current.setDate(current.getDate() + 1);
      }
    } else if (periodType === 'year') {
      // Mostra ogni mese dell'anno
      while (current <= endDate) {
        periods.push({
          label: current.toLocaleDateString('it-IT', { month: 'short' }),
          date: new Date(current)
        });
        current.setMonth(current.getMonth() + 1);
      }
    } else {
      // Per 'all' e 'custom', mostra per mese
      const monthStart = new Date(current.getFullYear(), current.getMonth(), 1);
      while (monthStart <= endDate) {
        periods.push({
          label: monthStart.toLocaleDateString('it-IT', { month: 'short' }),
          date: new Date(monthStart)
        });
        monthStart.setMonth(monthStart.getMonth() + 1);
      }
    }

    return periods;
  };

  const periods = generatePeriods();

  // Aggrega transazioni per periodo e categoria
  const timelineData = periods.map(period => {
    const periodTransactions = filteredTransactions.filter(t => {
      const tDate = new Date(t.date);

      if (periodType === 'year' || periodType === 'all' || periodType === 'custom') {
        // Confronta per mese
        return tDate.getMonth() === period.date.getMonth() &&
               tDate.getFullYear() === period.date.getFullYear();
      } else {
        // Confronta per giorno
        return tDate.getDate() === period.date.getDate() &&
               tDate.getMonth() === period.date.getMonth() &&
               tDate.getFullYear() === period.date.getFullYear();
      }
    });

    const categoryAmounts: Record<string, number> = {};
    let totalAmount = 0;

    periodTransactions.forEach(t => {
      const amount = Math.abs(t.amount);
      categoryAmounts[t.category] = (categoryAmounts[t.category] || 0) + amount;
      totalAmount += amount;
    });

    return {
      label: period.label,
      totalAmount,
      categoryAmounts
    };
  });

  // Trova il massimo totale per periodo per normalizzare le barre
  const maxPeriodAmount = Math.max(...timelineData.map(d => d.totalAmount), 1);

  // Mappa colori per categoria (ogni categoria ha un colore completamente diverso)
  const categoryColorMap = new Map<string, string>();
  // Palette di colori molto diversi tra loro per massima distinguibilità
  const baseColors = [
    '#ef4444', // rosso
    '#fb923c', // arancione chiaro
    '#fbbf24', // giallo oro
    '#84cc16', // lime
    '#10b981', // verde
    '#14b8a6', // teal
    '#06b6d4', // cyan
    '#3b82f6', // blu
    '#8b5cf6', // viola
    '#d946ef', // fucsia
    '#ec4899', // pink
    '#be123c', // rosso scuro
  ];

  categoryStats.forEach((cat, index) => {
    categoryColorMap.set(cat.name, baseColors[index % baseColors.length]);
  });

  const getCategoryColor = (categoryName: string) => {
    return categoryColorMap.get(categoryName) || baseColors[0];
  };

  // Calcola statistiche per sottocategorie di una categoria specifica
  const getSubcategoryStats = (categoryName: string) => {
    const categoryTransactions = filteredTransactions.filter(t => t.category === categoryName);
    const categoryTotal = categoryTransactions.reduce((sum, t) => sum + Math.abs(t.amount), 0);

    const subcategoryMap = new Map<string, { amount: number; count: number }>();

    categoryTransactions.forEach(t => {
      const subName = t.subcategory || 'Altro';
      const existing = subcategoryMap.get(subName);
      if (existing) {
        existing.amount += Math.abs(t.amount);
        existing.count += 1;
      } else {
        subcategoryMap.set(subName, {
          amount: Math.abs(t.amount),
          count: 1
        });
      }
    });

    const subcategoryStats = Array.from(subcategoryMap.entries()).map(([name, data]) => ({
      name,
      amount: data.amount,
      count: data.count,
      percentage: categoryTotal > 0 ? (data.amount / categoryTotal) * 100 : 0
    }));

    // Ordina per amount decrescente
    subcategoryStats.sort((a, b) => b.amount - a.amount);

    return subcategoryStats;
  };

  const toggleCategory = (categoryName: string) => {
    setExpandedCategory(expandedCategory === categoryName ? null : categoryName);
  };

  // Calcola l'andamento del saldo nel tempo
  const getBalanceTrendData = () => {
    // Trova l'ultima transazione nel periodo
    const lastTransactionDate = transactions.length > 0
      ? new Date(Math.max(...transactions.map(t => new Date(t.date).getTime())))
      : startDate;

    // Calcola il saldo iniziale (prima del periodo selezionato)
    const allTransactionsBeforeStart = transactions.filter(t => {
      const tDate = new Date(t.date);
      return tDate < startDate;
    });

    const initialBalance = allTransactionsBeforeStart.reduce((sum, t) => {
      if (t.type === 'income') {
        return sum + t.amount;
      } else if (t.type === 'expense') {
        return sum - Math.abs(t.amount);
      }
      return sum;
    }, 0);

    // Per ogni periodo, calcola il saldo progressivo
    let cumulativeBalance = initialBalance;

    const allPeriods = periods
      .filter(period => period.date <= lastTransactionDate)
      .map(period => {
        const periodTransactions = transactions.filter(t => {
          const tDate = new Date(t.date);

          if (periodType === 'year' || periodType === 'all' || periodType === 'custom') {
            return tDate.getMonth() === period.date.getMonth() &&
                   tDate.getFullYear() === period.date.getFullYear();
          } else {
            return tDate.getDate() === period.date.getDate() &&
                   tDate.getMonth() === period.date.getMonth() &&
                   tDate.getFullYear() === period.date.getFullYear();
          }
        });

        periodTransactions.forEach(t => {
          if (t.type === 'income') {
            cumulativeBalance += t.amount;
          } else if (t.type === 'expense') {
            cumulativeBalance -= Math.abs(t.amount);
          }
        });

        return {
          label: period.label,
          balance: cumulativeBalance,
          hasTransactions: periodTransactions.length > 0,
          date: period.date
        };
      });

    // Filtra solo i periodi con transazioni (più il primo se non ha transazioni)
    const balanceData = allPeriods.filter((period, index) =>
      period.hasTransactions || index === 0
    );

    return balanceData;
  };

  const balanceTrendData = getBalanceTrendData();
  const dataMaxBalance = Math.max(...balanceTrendData.map(d => d.balance));
  const dataMinBalance = Math.min(...balanceTrendData.map(d => d.balance));

  // Assicurati che lo 0 sia sempre incluso nel range
  const maxBalance = Math.max(dataMaxBalance, 0);
  const minBalance = Math.min(dataMinBalance, 0);
  const balanceRange = Math.max(maxBalance - minBalance, 1); // Evita divisione per zero

  console.log('Balance Trend Data:', {
    dataPoints: balanceTrendData.length,
    allData: balanceTrendData,
    minBalance,
    maxBalance,
    balanceRange,
    lastTransactionDate: transactions.length > 0
      ? new Date(Math.max(...transactions.map(t => new Date(t.date).getTime())))
      : null
  });

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-4">
        {/* Period Selector */}
        <PeriodSelector
          startDate={startDate}
          endDate={endDate}
          onPeriodChange={handlePeriodChange}
          onCustomClick={() => setIsDatePickerOpen(true)}
        />

        {/* Saldi periodo */}
        <div className="grid grid-cols-2 gap-3">
          <div className="card bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-green-200 dark:border-green-800">
            <div className="text-sm text-green-700 dark:text-green-300 mb-1">Entrate</div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {formatCurrency(totalIncome)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 border-red-200 dark:border-red-800">
            <div className="text-sm text-red-700 dark:text-red-300 mb-1">Uscite</div>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {formatCurrency(totalExpense)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border-blue-200 dark:border-blue-800">
            <div className="text-sm text-blue-700 dark:text-blue-300 mb-1">Investimenti</div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {formatCurrency(totalInvestment)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-purple-200 dark:border-purple-800">
            <div className="text-sm text-purple-700 dark:text-purple-300 mb-1">Saldo Periodo</div>
            <div className={`text-2xl font-bold ${
              periodBalance >= 0
                ? 'text-green-600 dark:text-green-400'
                : 'text-red-600 dark:text-red-400'
            }`}>
              {formatCurrency(periodBalance)}
            </div>
          </div>
        </div>

        {/* Filtri */}
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('expense')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filter === 'expense'
                ? 'bg-red-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            💸 Uscite
          </button>
          <button
            onClick={() => setFilter('income')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filter === 'income'
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            💰 Entrate
          </button>
          <button
            onClick={() => setFilter('investment')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filter === 'investment'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            📈 Investimenti
          </button>
        </div>

        {/* Messaggio quando non ci sono transazioni */}
        {filteredTransactions.length === 0 && (
          <div className="card text-center py-12">
            <div className="text-gray-500 dark:text-gray-400 text-lg">
              Non ci sono transazioni nel periodo considerato
            </div>
          </div>
        )}

        {/* Grafico temporale */}
        {filteredTransactions.length > 0 && (
          <div className="card">
            <div className="flex items-end justify-between gap-1 h-48">
              {timelineData.map((period, index) => {
                const barHeight = period.totalAmount > 0
                  ? (period.totalAmount / maxPeriodAmount) * 100
                  : 0;

                // Determina se mostrare la label
                const totalPeriods = timelineData.length;
                let showLabel = false;

                // Per i mesi (anno o custom lungo), mostra più label
                if (periodType === 'year' || (periodType === 'all' || periodType === 'custom') && totalPeriods <= 12) {
                  showLabel = true; // Mostra tutti i mesi
                } else if (totalPeriods <= 7) {
                  showLabel = true;
                } else if (totalPeriods <= 14) {
                  showLabel = index % 2 === 0 || index === totalPeriods - 1;
                } else if (totalPeriods <= 21) {
                  showLabel = index % 3 === 0 || index === totalPeriods - 1;
                } else if (totalPeriods <= 31) {
                  // Per i mesi, salta 2 invece di 5 (1, 4, 7, 10, 13...)
                  showLabel = index % 3 === 0 || index === totalPeriods - 1;
                } else {
                  showLabel = index % 7 === 0 || index === totalPeriods - 1;
                }

                // Mostra la barretta dove non c'è label (sia con che senza transazioni)
                const showPlaceholder = !showLabel;

                return (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    {/* Barra */}
                    <div className="w-full flex flex-col justify-end" style={{ height: '160px' }}>
                      {period.totalAmount > 0 ? (
                        <div
                          className="w-full rounded-t flex flex-col-reverse"
                          style={{ height: `${barHeight}%` }}
                        >
                          {categoryStats.map((cat) => {
                            const categoryAmount = period.categoryAmounts[cat.name] || 0;
                            const categoryPercentage = period.totalAmount > 0
                              ? (categoryAmount / period.totalAmount) * 100
                              : 0;

                            if (categoryPercentage === 0) return null;

                            return (
                              <div
                                key={cat.name}
                                style={{
                                  height: `${categoryPercentage}%`,
                                  backgroundColor: getCategoryColor(cat.name)
                                }}
                                title={`${cat.name}: ${formatCurrency(categoryAmount)}`}
                              />
                            );
                          })}
                        </div>
                      ) : showPlaceholder ? (
                        <div className="w-full h-1 bg-gray-300 dark:bg-gray-600 rounded" />
                      ) : null}
                    </div>
                    {/* Label - sempre h-4 per barre colorate, normale per placeholder */}
                    {period.totalAmount > 0 ? (
                      <div className="mt-1 h-4 flex items-end justify-center w-full">
                        {showLabel ? (
                          <span className="text-xs text-gray-600 dark:text-gray-400">{period.label}</span>
                        ) : (
                          <div className="w-full h-1 bg-gray-300 dark:bg-gray-600 rounded" />
                        )}
                      </div>
                    ) : showLabel ? (
                      <div className="text-xs text-gray-600 dark:text-gray-400 text-center mt-1">
                        {period.label}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Grafico a barre per categoria */}
        {categoryStats.length > 0 && (
          <div className="space-y-3">
            {categoryStats.map((stat) => {
              const isExpanded = expandedCategory === stat.name;
              const subcategoryStats = getSubcategoryStats(stat.name);
              // Ha sottocategorie reali solo se ce ne sono più di una, o se l'unica non è "Altro"
              const hasSubcategories = subcategoryStats.length > 1 ||
                (subcategoryStats.length === 1 && subcategoryStats[0].name !== 'Altro');

              return (
                <div key={stat.name} className="card">
                  {/* Header categoria - cliccabile solo se ha sottocategorie */}
                  <div
                    className={`flex items-center justify-between mb-2 ${hasSubcategories ? 'cursor-pointer' : ''}`}
                    onClick={hasSubcategories ? () => toggleCategory(stat.name) : undefined}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{stat.icon}</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {stat.name}
                      </span>
                      {hasSubcategories && (
                        <svg
                          width="12"
                          height="12"
                          viewBox="0 0 12 12"
                          fill="currentColor"
                          className="text-gray-400 dark:text-gray-500 flex-shrink-0"
                          style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
                        >
                          <path d="M4 2 L9 6 L4 10 Z" />
                        </svg>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-gray-900 dark:text-gray-100">
                        {formatCurrency(stat.amount)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {stat.count} trans.
                      </div>
                    </div>
                  </div>

                  {/* Barra di progresso */}
                  <div className="relative w-full h-8 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${stat.percentage}%`,
                        backgroundColor: getCategoryColor(stat.name)
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-end pr-3">
                      <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                        {stat.percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Sottocategorie espanse */}
                  {isExpanded && subcategoryStats.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
                      {subcategoryStats.map((substat) => (
                        <div key={substat.name}>
                          <div className="flex items-center justify-between mb-1 text-sm">
                            <span className="text-gray-700 dark:text-gray-300">
                              {substat.name}
                            </span>
                            <div className="text-right">
                              <span className="font-medium text-gray-900 dark:text-gray-100">
                                {formatCurrency(substat.amount)}
                              </span>
                              <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                                ({substat.count} trans.)
                              </span>
                            </div>
                          </div>
                          {/* Barra sottocategoria */}
                          <div className="relative w-full h-6 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
                            <div
                              className="h-full transition-all duration-500"
                              style={{
                                width: `${substat.percentage}%`,
                                backgroundColor: getCategoryColor(stat.name),
                                opacity: 0.7
                              }}
                            />
                            <div className="absolute inset-0 flex items-center justify-end pr-2">
                              <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                                {substat.percentage.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Grafico andamento saldo */}
        {transactions.length > 0 && (
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Andamento Saldo
            </h3>
            <div className="relative h-64">
              {/* Asse Y - labels */}
              <div className="absolute left-0 top-0 bottom-8 flex flex-col justify-between text-xs text-gray-500 dark:text-gray-400 pr-2">
                <span>{formatCurrency(maxBalance)}</span>
                <span>{formatCurrency((maxBalance + minBalance) / 2)}</span>
                <span>{formatCurrency(minBalance)}</span>
              </div>

              {/* Area grafico */}
              <div className="ml-16 mr-2 h-full flex flex-col">
                {/* Container per linea */}
                <div className="flex-1 relative mb-6">
                  {/* Griglia di sfondo */}
                  <div className="absolute inset-0 z-0">
                    {/* Linea dello zero */}
                    {minBalance < 0 && (
                      <div
                        className="absolute left-0 right-0 border-t border-gray-300 dark:border-gray-600"
                        style={{
                          bottom: `${balanceRange > 0 ? ((0 - minBalance) / balanceRange) * 100 : 50}%`
                        }}
                      />
                    )}
                  </div>

                  {/* SVG per la linea */}
                  <svg className="absolute inset-0 w-full h-full z-10" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <polyline
                      fill="none"
                      stroke="#9ca3af"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      vectorEffect="non-scaling-stroke"
                      points={balanceTrendData.map((point, index) => {
                        const totalPoints = balanceTrendData.length;
                        const x = totalPoints > 1
                          ? (index / (totalPoints - 1)) * 100
                          : 50;
                        const y = balanceRange > 0
                          ? 100 - ((point.balance - minBalance) / balanceRange) * 100
                          : 50;
                        return `${x},${y}`;
                      }).join(' ')}
                    />
                  </svg>

                  {/* Punti sui giorni con transazioni */}
                  {balanceTrendData.map((point, index) => {
                    // Mostra il punto se ci sono transazioni O se è il primo/ultimo punto
                    const isFirstOrLast = index === 0 || index === balanceTrendData.length - 1;
                    if (!point.hasTransactions && !isFirstOrLast) return null;

                    const totalPoints = balanceTrendData.length;
                    const xPercent = totalPoints > 1
                      ? (index / (totalPoints - 1)) * 100
                      : 50;
                    const yPercent = balanceRange > 0
                      ? 100 - ((point.balance - minBalance) / balanceRange) * 100
                      : 50;

                    return (
                      <div
                        key={index}
                        className="absolute w-3 h-3 rounded-full bg-white border-2 shadow-md transform -translate-x-1/2 -translate-y-1/2 hover:scale-125 transition-transform cursor-pointer z-20"
                        style={{
                          left: `${xPercent}%`,
                          top: `${yPercent}%`,
                          borderColor: point.balance < 0 ? '#ef4444' : '#10b981'
                        }}
                        title={`${point.label}: ${formatCurrency(point.balance)}`}
                      />
                    );
                  })}
                </div>

                {/* Asse X - labels */}
                <div className="relative h-5">
                  {balanceTrendData.map((point, index) => {
                    const totalPeriods = balanceTrendData.length;
                    let showLabel = false;

                    if (periodType === 'year' || (periodType === 'all' || periodType === 'custom') && totalPeriods <= 12) {
                      showLabel = true;
                    } else if (totalPeriods <= 7) {
                      showLabel = true;
                    } else if (totalPeriods <= 14) {
                      showLabel = index % 2 === 0 || index === totalPeriods - 1;
                    } else if (totalPeriods <= 21) {
                      showLabel = index % 3 === 0 || index === totalPeriods - 1;
                    } else if (totalPeriods <= 31) {
                      showLabel = index % 3 === 0 || index === totalPeriods - 1;
                    } else {
                      showLabel = index % 7 === 0 || index === totalPeriods - 1;
                    }

                    const xPosition = totalPeriods > 1
                      ? (index / (totalPeriods - 1)) * 100
                      : 50;

                    return showLabel ? (
                      <div
                        key={index}
                        className="absolute text-xs text-gray-600 dark:text-gray-400 font-medium transform -translate-x-1/2"
                        style={{ left: `${xPosition}%` }}
                      >
                        {point.label}
                      </div>
                    ) : null;
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Date Range Picker */}
        <DateRangePicker
          isOpen={isDatePickerOpen}
          onClose={() => setIsDatePickerOpen(false)}
          onConfirm={handleCustomPeriodConfirm}
          initialStart={startDate}
          initialEnd={endDate}
        />
      </div>
    </Layout>
  );
}
