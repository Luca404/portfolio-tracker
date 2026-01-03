import { useState } from 'react';
import Modal from './Modal';

interface DateRangePickerProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (startDate: Date, endDate: Date) => void;
  initialStart?: Date;
  initialEnd?: Date;
}

export default function DateRangePicker({
  isOpen,
  onClose,
  onConfirm,
  initialStart,
  initialEnd
}: DateRangePickerProps) {
  const [startDate, setStartDate] = useState(
    initialStart ? initialStart.toISOString().split('T')[0] : new Date().toISOString().split('T')[0]
  );
  const [endDate, setEndDate] = useState(
    initialEnd ? initialEnd.toISOString().split('T')[0] : new Date().toISOString().split('T')[0]
  );

  const handleConfirm = () => {
    const start = new Date(startDate);
    const end = new Date(endDate);
    end.setHours(23, 59, 59, 999);

    if (start > end) {
      alert('La data di inizio deve essere prima della data di fine');
      return;
    }

    onConfirm(start, end);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Seleziona Periodo">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Data Inizio
          </label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Data Fine
          </label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        <div className="flex gap-3 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Annulla
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="flex-1 px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors"
          >
            Conferma
          </button>
        </div>
      </div>
    </Modal>
  );
}
