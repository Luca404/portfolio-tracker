import { useState } from 'react';

interface MonthYearPickerProps {
  selectedMonth: number;  // 0-11 (gennaio = 0)
  selectedYear: number;
  onMonthChange: (month: number) => void;
  onYearChange: (year: number) => void;
  className?: string;
}

const MONTHS = [
  'Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
  'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic'
];

export default function MonthYearPicker({
  selectedMonth,
  selectedYear,
  onMonthChange,
  onYearChange,
  className = ''
}: MonthYearPickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);

  const handlePrevMonth = () => {
    if (selectedMonth === 0) {
      onMonthChange(11);
      onYearChange(selectedYear - 1);
    } else {
      onMonthChange(selectedMonth - 1);
    }
  };

  const handleNextMonth = () => {
    if (selectedMonth === 11) {
      onMonthChange(0);
      onYearChange(selectedYear + 1);
    } else {
      onMonthChange(selectedMonth + 1);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div className="flex items-center gap-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-300 dark:border-gray-600 px-2 py-1.5">
        <button
          type="button"
          onClick={handlePrevMonth}
          className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm"
        >
          ←
        </button>

        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="text-center text-sm font-medium text-gray-900 dark:text-gray-100 whitespace-nowrap px-1"
        >
          {MONTHS[selectedMonth]} {selectedYear}
        </button>

        <button
          type="button"
          onClick={handleNextMonth}
          className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm"
        >
          →
        </button>
      </div>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-50 mt-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-300 dark:border-gray-600 p-4 min-w-[280px]">
            {/* Anno selector */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Anno
              </label>
              <div className="grid grid-cols-3 gap-2">
                {years.map(year => (
                  <button
                    key={year}
                    type="button"
                    onClick={() => {
                      onYearChange(year);
                      setIsOpen(false);
                    }}
                    className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                      year === selectedYear
                        ? 'bg-primary-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {year}
                  </button>
                ))}
              </div>
            </div>

            {/* Mese selector */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Mese
              </label>
              <div className="grid grid-cols-4 gap-2">
                {MONTHS.map((month, index) => (
                  <button
                    key={month}
                    type="button"
                    onClick={() => {
                      onMonthChange(index);
                      setIsOpen(false);
                    }}
                    className={`px-2 py-2 rounded-lg text-sm font-medium transition-colors ${
                      index === selectedMonth
                        ? 'bg-primary-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {month}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
