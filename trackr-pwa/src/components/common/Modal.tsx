import { type ReactNode, useEffect } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  onBackdropClick?: () => void;
  title: string | ReactNode;
  children: ReactNode;
}

export default function Modal({ isOpen, onClose, onBackdropClick, title, children }: ModalProps) {
  useEffect(() => {
    if (isOpen) {
      // Blocca lo scroll della pagina quando il modal è aperto
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50"
        onClick={onBackdropClick || onClose}
      />

      {/* Modal Content */}
      <div
        className="relative bg-white dark:bg-gray-800 w-full sm:max-w-lg sm:rounded-lg rounded-t-2xl overflow-y-auto animate-slide-up"
        style={{
          maxHeight: '75vh',
          marginBottom: '5rem'
        }}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between z-10">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            {title}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4">
          {children}
        </div>
      </div>
    </div>
  );
}
