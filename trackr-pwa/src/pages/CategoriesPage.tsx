import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import type { CategoryWithStats, CategoryFormData, Subcategory, SubcategoryFormData } from '../types';

// Lista di icone disponibili
const AVAILABLE_ICONS = [
  '💰', '🍔', '🚗', '⚡', '🎮', '🏥', '🛍️', '🏠', '📱', '✈️',
  '🎬', '📚', '🎵', '🏋️', '☕', '🍕', '🚌', '💊', '👕', '🎁',
  '💳', '🎓', '🐶', '🌳', '🔧', '🖥️', '📸', '🎨', '⚽', '🍷'
];

type CategoryFilter = 'income' | 'expense' | 'investment';

export default function CategoriesPage() {
  const [categories, setCategories] = useState<CategoryWithStats[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [period, setPeriod] = useState<'month' | 'year'>('month');
  const [filter, setFilter] = useState<CategoryFilter>('expense');
  const [selectedCategory, setSelectedCategory] = useState<CategoryWithStats | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [isSubcategoryModal, setIsSubcategoryModal] = useState(false);

  useEffect(() => {
    loadCategories();
  }, [period]);

  const loadCategories = async () => {
    setIsLoading(true);
    try {
      const endDate = new Date();
      const startDate = new Date();

      if (period === 'month') {
        startDate.setDate(1);
      } else {
        startDate.setMonth(startDate.getMonth() - 11);
        startDate.setDate(1);
      }

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

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  const handleAddCategory = () => {
    setSelectedCategory(null);
    setIsEditMode(false);
    setIsSubcategoryModal(false);
    setIsModalOpen(true);
  };

  const handleEditCategory = (category: CategoryWithStats) => {
    setSelectedCategory(category);
    setIsEditMode(true);
    setIsSubcategoryModal(false);
    setIsModalOpen(true);
  };

  const handleDeleteCategory = async (id: number) => {
    if (!confirm('Sei sicuro di voler eliminare questa categoria?')) return;
    try {
      await apiService.deleteCategory(id);
      await loadCategories();
    } catch (error) {
      console.error('Errore eliminazione categoria:', error);
      alert('Errore durante l\'eliminazione della categoria');
    }
  };

  const handleViewSubcategories = (category: CategoryWithStats) => {
    setSelectedCategory(category);
    setIsSubcategoryModal(false);
    setIsModalOpen(false);
  };

  const handleAddSubcategory = (category: CategoryWithStats) => {
    setSelectedCategory(category);
    setIsEditMode(false);
    setIsSubcategoryModal(true);
    setIsModalOpen(true);
  };

  const handleBack = () => {
    setSelectedCategory(null);
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  // Vista dettaglio categoria (sottocategorie)
  const isDetailView = selectedCategory && !isSubcategoryModal && !isEditMode;

  if (isDetailView) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-4">
            <button
              onClick={handleBack}
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
            >
              ← Indietro
            </button>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <span className="text-3xl">{selectedCategory.icon}</span>
              {selectedCategory.name}
            </h1>
          </div>

          {/* Sottocategorie */}
          <div className="space-y-3">
            {selectedCategory.subcategories.length === 0 ? (
              <div className="card text-center py-8 text-gray-500 dark:text-gray-400">
                Nessuna sottocategoria. Tocca il + per aggiungerne una!
              </div>
            ) : (
              selectedCategory.subcategories.map((sub) => (
                <SubcategoryCard
                  key={sub.id}
                  subcategory={sub}
                  categoryId={selectedCategory.id}
                  onUpdate={loadCategories}
                />
              ))
            )}
          </div>

          <FAB onClick={() => handleAddSubcategory(selectedCategory)} />

          {/* Modal per sottocategoria */}
          <Modal
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            title="Nuova Sottocategoria"
          >
            <SubcategoryForm
              categoryId={selectedCategory.id}
              onSuccess={() => {
                setIsModalOpen(false);
                loadCategories();
              }}
              onCancel={() => setIsModalOpen(false)}
            />
          </Modal>
        </div>
      </Layout>
    );
  }

  // Filtro categorie
  const filteredCategories = categories.filter(category => {
    return category.category_type === filter || (!category.category_type && filter === 'expense');
  });

  // Vista principale (lista categorie)
  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Categorie
          </h1>
          <div className="flex bg-gray-200 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setPeriod('month')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                period === 'month'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              Mese
            </button>
            <button
              onClick={() => setPeriod('year')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                period === 'year'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              Anno
            </button>
          </div>
        </div>

        {/* Filtri tipologia */}
        <div className="flex gap-2 overflow-x-auto">
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

        {/* Lista categorie */}
        <div className="space-y-3">
          {filteredCategories.length === 0 ? (
            <div className="card text-center py-8 text-gray-500 dark:text-gray-400">
              Nessuna categoria in questo filtro.
            </div>
          ) : (
            filteredCategories.map((category) => (
              <CategoryCard
                key={category.id}
                category={category}
                onEdit={handleEditCategory}
                onDelete={handleDeleteCategory}
                onViewSubcategories={handleViewSubcategories}
              />
            ))
          )}
        </div>

        <FAB onClick={handleAddCategory} />

        {/* Modal per categoria */}
        <Modal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          title={isSubcategoryModal
            ? 'Nuova Sottocategoria'
            : (isEditMode ? 'Modifica Categoria' : 'Nuova Categoria')
          }
        >
          {isSubcategoryModal ? (
            <SubcategoryForm
              categoryId={selectedCategory!.id}
              onSuccess={() => {
                setIsModalOpen(false);
                loadCategories();
              }}
              onCancel={() => setIsModalOpen(false)}
            />
          ) : (
            <CategoryForm
              category={isEditMode ? selectedCategory : null}
              onSuccess={() => {
                setIsModalOpen(false);
                loadCategories();
              }}
              onCancel={() => setIsModalOpen(false)}
            />
          )}
        </Modal>
      </div>
    </Layout>
  );
}

// Card per categoria
function CategoryCard({
  category,
  onEdit,
  onDelete,
  onViewSubcategories,
}: {
  category: CategoryWithStats;
  onEdit: (category: CategoryWithStats) => void;
  onDelete: (id: number) => void;
  onViewSubcategories: (category: CategoryWithStats) => void;
}) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  return (
    <div
      className="card cursor-pointer hover:shadow-lg transition-shadow"
      onClick={() => onViewSubcategories(category)}
    >
      <div className="flex items-center gap-4">
        <div className="text-4xl">{category.icon}</div>
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">
            {category.name}
          </h3>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {category.transaction_count} transazioni • {category.subcategories.length} sottocategorie
          </div>
        </div>
        <div className="text-right">
          <div className="font-bold text-lg text-gray-900 dark:text-gray-100">
            {formatCurrency(category.total_amount)}
          </div>
        </div>
      </div>

      <div className="flex gap-2 mt-4 justify-end">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onEdit(category);
          }}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          ✏️
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(category.id);
          }}
          className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
        >
          🗑️
        </button>
      </div>
    </div>
  );
}

// Card per sottocategoria
function SubcategoryCard({
  subcategory,
  categoryId,
  onUpdate,
}: {
  subcategory: Subcategory;
  categoryId: number;
  onUpdate: () => void;
}) {
  const [isEditing, setIsEditing] = useState(false);

  const handleDelete = async () => {
    if (!confirm('Sei sicuro di voler eliminare questa sottocategoria?')) return;
    try {
      await apiService.deleteSubcategory(categoryId, subcategory.id);
      await onUpdate();
    } catch (error) {
      console.error('Errore eliminazione sottocategoria:', error);
      alert('Errore durante l\'eliminazione della sottocategoria');
    }
  };

  if (isEditing) {
    return (
      <SubcategoryForm
        categoryId={categoryId}
        subcategory={subcategory}
        onSuccess={() => {
          setIsEditing(false);
          onUpdate();
        }}
        onCancel={() => setIsEditing(false)}
      />
    );
  }

  return (
    <div className="card flex items-center gap-4">
      <div className="text-3xl">{subcategory.icon}</div>
      <div className="flex-1">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
          {subcategory.name}
        </h3>
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => setIsEditing(true)}
          className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
        >
          ✏️
        </button>
        <button
          onClick={handleDelete}
          className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          🗑️
        </button>
      </div>
    </div>
  );
}

// Form per categoria
function CategoryForm({
  category,
  onSuccess,
  onCancel,
}: {
  category: CategoryWithStats | null;
  onSuccess: () => void;
  onCancel: () => void;
}) {
  const [formData, setFormData] = useState<CategoryFormData>({
    name: category?.name || '',
    icon: category?.icon || '💰',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (category) {
        await apiService.updateCategory(category.id, formData);
      } else {
        await apiService.createCategory(formData);
      }
      onSuccess();
    } catch (error: any) {
      console.error('Errore salvataggio categoria:', error);
      const errorMsg = error?.response?.data?.detail || error?.message || 'Errore durante il salvataggio della categoria';
      alert(errorMsg);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Nome Categoria
        </label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          className="input-field"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Icona
        </label>
        <div className="grid grid-cols-10 gap-2">
          {AVAILABLE_ICONS.map((icon) => (
            <button
              key={icon}
              type="button"
              onClick={() => setFormData({ ...formData, icon })}
              className={`text-2xl p-2 rounded transition-colors flex items-center justify-center ${
                formData.icon === icon
                  ? 'bg-primary-500 dark:bg-primary-600'
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {icon}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg hover:bg-gray-400 dark:hover:bg-gray-600 transition-colors"
        >
          Annulla
        </button>
        <button
          type="submit"
          className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          {category ? 'Salva' : 'Crea'}
        </button>
      </div>
    </form>
  );
}

// Form per sottocategoria
function SubcategoryForm({
  categoryId,
  subcategory,
  onSuccess,
  onCancel,
}: {
  categoryId: number;
  subcategory?: Subcategory;
  onSuccess: () => void;
  onCancel: () => void;
}) {
  const [formData, setFormData] = useState<SubcategoryFormData>({
    name: subcategory?.name || '',
    icon: subcategory?.icon || '📌',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (subcategory) {
        await apiService.updateSubcategory(categoryId, subcategory.id, formData);
      } else {
        await apiService.createSubcategory(categoryId, formData);
      }
      onSuccess();
    } catch (error: any) {
      console.error('Errore salvataggio sottocategoria:', error);
      const errorMsg = error?.response?.data?.detail || error?.message || 'Errore durante il salvataggio della sottocategoria';
      alert(errorMsg);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Nome Sottocategoria
        </label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          className="input-field"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Icona
        </label>
        <div className="grid grid-cols-10 gap-2">
          {AVAILABLE_ICONS.map((icon) => (
            <button
              key={icon}
              type="button"
              onClick={() => setFormData({ ...formData, icon })}
              className={`text-2xl p-2 rounded transition-colors flex items-center justify-center ${
                formData.icon === icon
                  ? 'bg-primary-500 dark:bg-primary-600'
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {icon}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg hover:bg-gray-400 dark:hover:bg-gray-600 transition-colors"
        >
          Annulla
        </button>
        <button
          type="submit"
          className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          {subcategory ? 'Salva' : 'Crea'}
        </button>
      </div>
    </form>
  );
}
