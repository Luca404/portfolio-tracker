import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PeriodSelector from '../components/common/PeriodSelector';
import DateRangePicker from '../components/common/DateRangePicker';
import { usePeriod } from '../hooks/usePeriod';
import type { CategoryWithStats, CategoryFormData, SubcategoryFormData } from '../types';

type CategoryFilter = 'income' | 'expense' | 'investment';
type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

const CATEGORY_ICONS = [
  '🍔', '🚗', '⚡', '🎮', '🏥', '🛍️', '💰', '💵', '📌',
  '🎬', '📚', '🎵', '🏋️', '☕', '🍕', '🚌', '💊', '👕', '🎁',
  '💳', '🎓', '🐶', '🌳', '🔧', '🖥️', '📸', '🎨', '⚽', '🍷'
];

export default function CategoriesPageNew() {
  const [categories, setCategories] = useState<CategoryWithStats[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<CategoryFilter>('expense');
  const [selectedCategory, setSelectedCategory] = useState<CategoryWithStats | null>(null);
  const [isSubcategoryModalOpen, setIsSubcategoryModalOpen] = useState(false);
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false);
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  // Period state - condiviso tra le pagine
  const { startDate, endDate, setPeriod } = usePeriod();

  const [categoryFormData, setCategoryFormData] = useState<CategoryFormData>({
    name: '',
    icon: '📌',
    category_type: 'expense',
  });

  const [subcategoryFormData, setSubcategoryFormData] = useState<SubcategoryFormData>({
    name: '',
    icon: '📌',
  });

  useEffect(() => {
    loadCategories();
  }, [startDate, endDate]);

  const loadCategories = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getCategories({
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0],
      });
      setCategories(data);
    } catch (error) {
      console.error('Errore caricamento categorie:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePeriodChange = (start: Date, end: Date, type: PeriodType) => {
    setPeriod(start, end, type);
  };

  const handleCustomPeriodConfirm = (start: Date, end: Date) => {
    setPeriod(start, end, 'custom');
  };

  const handleCategoryClick = (category: CategoryWithStats) => {
    setSelectedCategory(category);
    setIsSubcategoryModalOpen(true);
  };

  const handleEditCategory = (e: React.MouseEvent, category: CategoryWithStats) => {
    e.stopPropagation();
    setSelectedCategory(category);
    setIsEditMode(true);
    setCategoryFormData({
      name: category.name,
      icon: category.icon,
      category_type: category.category_type,
    });
    setIsCategoryModalOpen(true);
  };

  const handleDeleteCategory = async (e: React.MouseEvent, categoryId: number) => {
    e.stopPropagation();
    if (confirm('Sei sicuro di voler eliminare questa categoria?')) {
      try {
        await apiService.deleteCategory(categoryId);
        await loadCategories();
        setIsSubcategoryModalOpen(false);
      } catch (error) {
        console.error('Errore eliminazione categoria:', error);
      }
    }
  };

  const handleCategorySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (isEditMode && selectedCategory) {
        await apiService.updateCategory(selectedCategory.id, categoryFormData);
      } else {
        await apiService.createCategory(categoryFormData);
      }
      await loadCategories();
      setIsCategoryModalOpen(false);
      setIsEditMode(false);
    } catch (error) {
      console.error('Errore salvataggio categoria:', error);
    }
  };

  const handleSubcategorySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCategory) return;

    try {
      await apiService.createSubcategory(selectedCategory.id, subcategoryFormData);
      await loadCategories();
      setSubcategoryFormData({ name: '', icon: '📌' });
      const updatedCategories = await apiService.getCategories({
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0],
      });
      const updated = updatedCategories.find(c => c.id === selectedCategory.id);
      if (updated) setSelectedCategory(updated);
    } catch (error) {
      console.error('Errore creazione sottocategoria:', error);
    }
  };

  const handleDeleteSubcategory = async (subcategoryId: number) => {
    if (!selectedCategory) return;
    if (confirm('Sei sicuro di voler eliminare questa sottocategoria?')) {
      try {
        await apiService.deleteSubcategory(selectedCategory.id, subcategoryId);
        await loadCategories();
        const updatedCategories = await apiService.getCategories({
          start_date: startDate.toISOString().split('T')[0],
          end_date: endDate.toISOString().split('T')[0],
        });
        const updated = updatedCategories.find(c => c.id === selectedCategory.id);
        if (updated) setSelectedCategory(updated);
      } catch (error) {
        console.error('Errore eliminazione sottocategoria:', error);
      }
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  const filteredCategories = categories.filter(category => {
    return category.category_type === filter || (!category.category_type && filter === 'expense');
  });

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Categorie
        </h1>

        {/* Period Selector */}
        <PeriodSelector
          startDate={startDate}
          endDate={endDate}
          onPeriodChange={handlePeriodChange}
          onCustomClick={() => setIsDatePickerOpen(true)}
        />

        {/* Filtri tipologia */}
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

        {/* Grid categorie compatto */}
        {filteredCategories.length === 0 ? (
          <div className="card text-center py-12">
            <div className="text-gray-500 dark:text-gray-400">
              Nessuna categoria in questo filtro.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            {filteredCategories.map((category) => (
              <button
                key={category.id}
                onClick={() => handleCategoryClick(category)}
                className="flex flex-col items-center p-3 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 hover:shadow-md transition-all"
              >
                <div className="text-3xl mb-1">{category.icon}</div>
                <div className="text-xs font-medium text-gray-900 dark:text-gray-100 text-center line-clamp-1 w-full">
                  {category.name}
                </div>
                <div className={`text-sm font-bold mt-1 ${
                  filter === 'expense'
                    ? 'text-red-600 dark:text-red-400'
                    : filter === 'income'
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-blue-600 dark:text-blue-400'
                }`}>
                  {formatCurrency(Math.abs(category.total_amount))}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {category.transaction_count} trans.
                </div>
              </button>
            ))}
          </div>
        )}

        {/* FAB */}
        <FAB onClick={() => {
          setIsEditMode(false);
          setCategoryFormData({ name: '', icon: '📌', category_type: filter });
          setIsCategoryModalOpen(true);
        }} />

        {/* Modal sottocategorie */}
        <Modal
          isOpen={isSubcategoryModalOpen}
          onClose={() => setIsSubcategoryModalOpen(false)}
          title={
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{selectedCategory?.icon}</span>
                <span>{selectedCategory?.name}</span>
              </div>
              <button
                onClick={(e) => selectedCategory && handleEditCategory(e, selectedCategory)}
                className="text-gray-600 dark:text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 text-xl"
              >
                ⚙️
              </button>
            </div>
          }
        >
          <div className="space-y-4">
            {/* Lista sottocategorie */}
            {selectedCategory && selectedCategory.subcategories && selectedCategory.subcategories.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Sottocategorie
                </h4>
                <div className="space-y-2">
                  {selectedCategory.subcategories.map((sub) => (
                    <div
                      key={sub.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xl">{sub.icon}</span>
                        <span className="text-sm text-gray-900 dark:text-gray-100">{sub.name}</span>
                      </div>
                      <button
                        onClick={() => handleDeleteSubcategory(sub.id)}
                        className="text-red-500 hover:text-red-700 text-sm"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Form nuova sottocategoria */}
            <form onSubmit={handleSubcategorySubmit} className="space-y-3">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Nuova Sottocategoria
              </h4>

              <input
                type="text"
                value={subcategoryFormData.name}
                onChange={(e) => setSubcategoryFormData({ ...subcategoryFormData, name: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
                placeholder="Nome sottocategoria"
                required
              />

              <div className="grid grid-cols-6 gap-2">
                {CATEGORY_ICONS.slice(0, 12).map((icon) => (
                  <button
                    key={icon}
                    type="button"
                    onClick={() => setSubcategoryFormData({ ...subcategoryFormData, icon })}
                    className={`p-2 text-xl rounded-lg border-2 transition-colors ${
                      subcategoryFormData.icon === icon
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700'
                    }`}
                  >
                    {icon}
                  </button>
                ))}
              </div>

              <button
                type="submit"
                className="w-full px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors"
              >
                Aggiungi
              </button>
            </form>
          </div>
        </Modal>

        {/* Modal categoria */}
        <Modal
          isOpen={isCategoryModalOpen}
          onClose={() => setIsCategoryModalOpen(false)}
          title={isEditMode ? 'Modifica Categoria' : 'Nuova Categoria'}
        >
          <form onSubmit={handleCategorySubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Nome
              </label>
              <input
                type="text"
                value={categoryFormData.name}
                onChange={(e) => setCategoryFormData({ ...categoryFormData, name: e.target.value })}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Icona
              </label>
              <div className="grid grid-cols-6 gap-2">
                {CATEGORY_ICONS.map((icon) => (
                  <button
                    key={icon}
                    type="button"
                    onClick={() => setCategoryFormData({ ...categoryFormData, icon })}
                    className={`p-3 text-2xl rounded-lg border-2 transition-colors ${
                      categoryFormData.icon === icon
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700'
                    }`}
                  >
                    {icon}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              {isEditMode && selectedCategory && (
                <button
                  type="button"
                  onClick={(e) => handleDeleteCategory(e, selectedCategory.id)}
                  className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors"
                >
                  Elimina
                </button>
              )}
              <button
                type="button"
                onClick={() => setIsCategoryModalOpen(false)}
                className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Annulla
              </button>
              <button
                type="submit"
                className="flex-1 px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600"
              >
                {isEditMode ? 'Salva' : 'Crea'}
              </button>
            </div>
          </form>
        </Modal>

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
