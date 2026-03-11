import { useState, useMemo } from 'react';
import { apiService } from '../services/api';
import { useData } from '../contexts/DataContext';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import ConfirmDialog from '../components/common/ConfirmDialog';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PeriodSelector from '../components/common/PeriodSelector';
import DateRangePicker from '../components/common/DateRangePicker';
import { usePeriod } from '../hooks/usePeriod';
import type { CategoryWithStats, CategoryFormData, SubcategoryFormData } from '../types';

type CategoryFilter = 'income' | 'expense' | 'investment';
type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

const CATEGORY_ICONS = [
  '🍔', '🚌', '⚡', '🎮', '🏥', '🛍️', '💰', '💵', '📌',
  '🎬', '📚', '🎵', '🏋️', '☕', '🍕', '💊', '👕', '🎁',
  '💳', '🎓', '🐶', '🌳', '🔧', '🖥️', '📸', '🎨', '⚽', '🍷',
  '🏠', '🔑', '🚰', '💡', '📱', '🌐', '✈️', '🏖️', '🎭', '🎪',
  '🎯', '🎲', '🎰', '🎸', '🎹', '🎺', '🎻', '🥁', '🎤', '🎧',
  '📺', '📻', '📞', '☎️', '📠', '💻', '⌨️', '🖱️', '🖨️', '💾',
  '🍎', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🍒', '🍑', '🥝',
  '🥑', '🍆', '🥒', '🥕', '🌽', '🥔', '🧅', '🧄', '🥖', '🥐',
  '🍞', '🥨', '🧀', '🥚', '🍳', '🥓', '🥩', '🍗', '🍖', '🌭',
  '🍟', '🌮', '🌯', '🥙', '🥗', '🍝', '🍜', '🍲',
  '🍛', '🍣', '🍱', '🥟', '🍤', '🍙', '🍚', '🍘', '🍥', '🥠',
  '🍢', '🍡', '🍧', '🍨', '🍦', '🥧', '🍰', '🎂', '🍮', '🍭',
  '🍬', '🍫', '🍿', '🍩', '🍪', '🌰', '🥜', '🍯', '🥛', '🍼',
  '🍵', '🧃', '🥤', '🍶', '🍺', '🍻', '🥂', '🥃',
  '🍸', '🍹', '🍾', '🧉', '🏀', '🏈', '⚾', '🥎', '🎾',
  '🏐', '🏉', '🥏', '🎱', '🏓', '🏸', '🏒', '🏑', '🥍', '🏏',
  '⛳', '🏹', '🎣', '🥊', '🥋', '🎽', '⛸️', '🥌', '🛷', '🎿',
  '⛷️', '🏂', '🤼', '🤸', '🤾', '🧗', '🚴', '🚵', '🧘',
  '🏃', '🚶', '💃', '🕺', '🤺', '🏇', '🏊', '🤽', '🚣', '🧜',
  '🚆', '🚇', '🚕', '🏍️', '🚲', '🚗', '🚙', '🚐', '🚛', '🚚',
];

// Mappa parole chiave -> icone suggerite
const ICON_SUGGESTIONS: Record<string, string[]> = {
  'cibo|food|mangiare|alimentari|ristorante|pranzo|cena|colazione|spuntino|snack|hamburger|pizza|pasta|sushi|insalata|taco|ramen|curry': ['🍔', '🍕', '🍝', '🍱', '🥗', '🌮', '🍜', '🍛', '🍗', '🍖', '🌭', '🍟', '🌯', '🥙', '🍲', '🍣', '🥟', '🍤'],
  'bevande|drink|bar|caffe|caffè|birra|vino|cocktail|bibita|te|tè|succo|acqua|bevanda': ['☕', '🍺', '🍷', '🥤', '🧃', '🍵', '🍹', '🍸', '🍶', '🍻', '🥂', '🥃', '🍾', '🧉'],
  'trasporti|auto|macchina|car|bus|metro|treno|viaggio|benzina|carburante|taxi|moto|bici|aereo|nave': ['🚌', '🚆', '🚇', '🚕', '🏍️', '🚲', '✈️', '🚗', '🚙', '🚐', '🚛', '🚚'],
  'casa|home|affitto|mutuo|bollette|acqua|luce|gas|riscaldamento|elettricità|immobiliare': ['🏠', '🔑', '💡', '🚰', '⚡', '🌡️'],
  'salute|medico|farmacia|ospedale|dottore|medicina|visita|analisi|dentista|oculista': ['🏥', '💊', '🩺', '💉', '🧬', '🧪'],
  'shopping|acquisti|negozio|abbigliamento|vestiti|moda|scarpe|accessori|abbigliamento|clothes': ['🛍️', '👕', '👗', '👠', '🎽', '🧥'],
  'intrattenimento|film|cinema|teatro|spettacolo|concerti|eventi|show|arte|cultura': ['🎬', '🎭', '🎪', '🎨', '🎤', '🎧'],
  'sport|palestra|fitness|allenamento|gym|calcio|basket|tennis|nuoto|corsa|yoga|ciclismo': ['🏋️', '⚽', '🏀', '🎾', '🏊', '🚴', '🧘', '🏈', '⚾', '🥎', '🏐', '🏉', '🏓', '🏸', '🏒'],
  'educazione|scuola|università|studio|libri|corso|formazione|learning|istruzione|lezioni': ['🎓', '📚', '✏️', '📖', '🎒', '📝'],
  'tecnologia|tech|computer|telefono|elettronica|pc|smartphone|tablet|software|hardware|internet|web': ['💻', '📱', '🖥️', '⌨️', '🖱️', '📡', '🖨️', '💾', '📞', '☎️', '📠'],
  'giochi|gaming|videogiochi|game|console|playstation|xbox|nintendo|gioco|scommesse|lotteria': ['🎮', '🎯', '🎲', '🎰', '🃏', '🎱'],
  'viaggi|viaggio|vacanza|aereo|hotel|turismo|volo|destinazione|weekend|ferie': ['✈️', '🏖️', '🗺️', '🧳', '🏨', '🎒'],
  'musica|music|concerti|strumenti|chitarra|piano|batteria|canzoni|spotify|artista': ['🎵', '🎸', '🎹', '🎤', '🎧', '🎺', '🥁', '🎻'],
  'animali|pet|cane|gatto|dog|cat|veterinario|animale|domestico|cucciolo': ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻'],
  'natura|piante|giardino|garden|fiori|verde|parco|ambiente|outdoor': ['🌳', '🌲', '🌿', '🍀', '🌻', '🌺', '🌸'],
  'lavoro|work|ufficio|office|business|riunione|meeting|progetto|stipendio|azienda': ['💼', '👔', '🏢', '📊', '📈', '💻', '📞'],
  'regali|gift|compleanno|birthday|festa|party|celebrazione|anniversario|presente': ['🎁', '🎈', '🎂', '🎉', '🎊', '🎀'],
  'bellezza|beauty|parrucchiere|estetica|makeup|trucco|cura|spa|benessere|massaggio': ['💄', '💅', '💇', '🧖', '💆', '🧴'],
  'pulizia|cleaning|detersivi|lavanderia|casa|igiene|sapone|lavatrice': ['🧹', '🧺', '🧼', '🧽', '🧴', '🚿'],
  'soldi|denaro|money|cash|banca|bank|finanza|risparmio|investimenti|portafoglio|euro|dollaro': ['💰', '💵', '💳', '🏦', '💸', '💶', '💷'],
  'frutta|fruit|mela|banana|arancia|limone|fragola|uva|kiwi|pesca': ['🍎', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🍒', '🍑', '🥝'],
  'verdura|vegetable|insalata|pomodoro|carota|patata|melanzana|cipolla|aglio': ['🥑', '🍆', '🥒', '🥕', '🌽', '🥔', '🧅', '🧄'],
  'dolci|dessert|torta|gelato|cioccolato|biscotti|caramelle|sweet': ['🍰', '🎂', '🍮', '🍭', '🍬', '🍫', '🍿', '🍩', '🍪', '🍧', '🍨', '🍦', '🥧'],
  'colazione|breakfast|pane|cornetto|brioche|croissant|cereali|latte': ['🥖', '🥐', '🍞', '🥨', '🧀', '🥚', '🍳', '🥓', '🥛', '🍼'],
  'energia|elettricità|power|corrente|batteria': ['⚡', '💡', '🔋'],
  'comunicazione|telefono|chiamata|messaggio|email|posta': ['📱', '📞', '☎️', '📠', '📺', '📻'],
  'foto|fotografia|camera|foto|immagine|picture': ['📸', '📷'],
  'strumenti|tools|attrezzi|riparazione|manutenzione': ['🔧', '🔨', '🛠️'],
  'divertimento|fun|gioco|entertainment': ['🎮', '🎯', '🎲', '🎰', '🎪'],
  'abbonamenti|abbonamento|subscription|mensile|annuale|netflix|spotify|prime|streaming': ['📱', '💻', '📺', '🎵', '🎬', '📡', '🌐', '💳'],
};

const getSuggestedIcons = (name: string): string[] => {
  if (!name || name.trim().length === 0) {
    return CATEGORY_ICONS;
  }

  const nameLower = name.toLowerCase().trim();

  // Se il nome è troppo corto (< 2 caratteri), mostra tutte le icone
  if (nameLower.length < 2) {
    return CATEGORY_ICONS;
  }

  // Array per memorizzare le corrispondenze con punteggio
  const matches: { keywords: string; icons: string[]; score: number }[] = [];

  for (const [keywords, icons] of Object.entries(ICON_SUGGESTIONS)) {
    const keywordList = keywords.split('|');
    let bestScore = 0;

    for (const keyword of keywordList) {
      const keywordLower = keyword.toLowerCase();

      // Corrispondenza esatta: punteggio massimo
      if (nameLower === keywordLower) {
        bestScore = Math.max(bestScore, 100);
      }
      // Il nome inizia con la keyword: punteggio alto
      else if (nameLower.startsWith(keywordLower)) {
        bestScore = Math.max(bestScore, 80);
      }
      // La keyword inizia con il nome (almeno 2 caratteri): punteggio medio-alto
      else if (keywordLower.startsWith(nameLower) && nameLower.length >= 2) {
        bestScore = Math.max(bestScore, 70);
      }
      // Il nome contiene la keyword all'inizio di una parola
      else if (nameLower.includes(' ' + keywordLower) || nameLower.includes('-' + keywordLower)) {
        bestScore = Math.max(bestScore, 50);
      }
    }

    if (bestScore > 0) {
      matches.push({ keywords, icons, score: bestScore });
    }
  }

  // Se non ci sono corrispondenze abbastanza forti, mostra tutte le icone nell'ordine originale
  if (matches.length === 0 || matches[0].score < 50) {
    return CATEGORY_ICONS;
  }

  // Ordina per punteggio decrescente
  matches.sort((a, b) => b.score - a.score);

  // Prendi solo le corrispondenze con punteggio >= 50
  const bestMatches = matches.filter(m => m.score >= 50);

  // Raccogli tutte le icone suggerite rimuovendo duplicati
  const suggestedIcons = new Set<string>();
  bestMatches.forEach(match => {
    match.icons.forEach(icon => suggestedIcons.add(icon));
  });

  // Aggiungi le restanti icone alla fine
  const remainingIcons = CATEGORY_ICONS.filter(icon => !suggestedIcons.has(icon));

  // Restituisci prima le icone suggerite, poi tutte le altre
  return [...Array.from(suggestedIcons), ...remainingIcons];
};

export default function CategoriesPage() {
  const { categories: baseCategories, transactions: allTransactions, isLoading, addCategory, updateCategory: updateCategoryCache, deleteCategory: deleteCategoryCache } = useData();
  const [filter, setFilter] = useState<CategoryFilter>('expense');
  const [selectedCategory, setSelectedCategory] = useState<CategoryWithStats | null>(null);
  const [isSubcategoryModalOpen, setIsSubcategoryModalOpen] = useState(false);
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false);
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  // Period state - condiviso tra le pagine
  const { startDate, endDate, setPeriod } = usePeriod();

  // Calcola le statistiche localmente dalle transazioni filtrate per periodo
  const categories = useMemo(() => {
    // Filtra le transazioni per il periodo
    const filteredTransactions = allTransactions.filter(transaction => {
      const transactionDate = new Date(transaction.date);
      return transactionDate >= startDate && transactionDate <= endDate;
    });

    // Crea una mappa per aggregare le statistiche per categoria
    const categoryStats = new Map<number, CategoryWithStats>();

    // Inizializza le categorie base
    baseCategories.forEach(category => {
      categoryStats.set(category.id, {
        ...category,
        total_amount: 0,
        transaction_count: 0,
        subcategories: category.subcategories?.map(sub => ({
          ...sub,
          total_amount: 0,
          transaction_count: 0,
        })) || [],
      });
    });

    // Aggrega le transazioni
    filteredTransactions.forEach(transaction => {
      const category = baseCategories.find(c => c.name === transaction.category);
      if (!category) return;

      // Filtra solo le transazioni che corrispondono al tipo della categoria
      // Se la categoria ha category_type null, conta tutte le transazioni
      if (category.category_type && transaction.type !== category.category_type) {
        return;
      }

      const stats = categoryStats.get(category.id);
      if (!stats) return;

      // Aggiorna totale categoria
      stats.total_amount += Math.abs(transaction.amount);
      stats.transaction_count += 1;

      // Aggiorna sottocategoria se presente
      if (transaction.subcategory && stats.subcategories) {
        const subcat = stats.subcategories.find(s => s.name === transaction.subcategory);
        if (subcat) {
          subcat.total_amount = (subcat.total_amount || 0) + Math.abs(transaction.amount);
          subcat.transaction_count = (subcat.transaction_count || 0) + 1;
        }
      }
    });

    return Array.from(categoryStats.values());
  }, [baseCategories, allTransactions, startDate, endDate]);

  const [categoryFormData, setCategoryFormData] = useState<CategoryFormData>({
    name: '',
    icon: '📌',
    category_type: 'expense',
  });

  const [subcategoryFormData, setSubcategoryFormData] = useState<SubcategoryFormData>({
    name: '',
  });

  const [showSubcategoryForm, setShowSubcategoryForm] = useState(false);

  // Confirm dialog states
  const [isDeleteCategoryDialogOpen, setIsDeleteCategoryDialogOpen] = useState(false);
  const [isDeleteSubcategoryDialogOpen, setIsDeleteSubcategoryDialogOpen] = useState(false);
  const [categoryToDelete, setCategoryToDelete] = useState<number | null>(null);
  const [subcategoryToDelete, setSubcategoryToDelete] = useState<number | null>(null);

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

  const handleDeleteCategory = (e: React.MouseEvent, categoryId: number) => {
    e.stopPropagation();
    setCategoryToDelete(categoryId);
    setIsDeleteCategoryDialogOpen(true);
  };

  const confirmDeleteCategory = async () => {
    if (categoryToDelete !== null) {
      try {
        await apiService.deleteCategory(categoryToDelete);
        deleteCategoryCache(categoryToDelete);
        setIsCategoryModalOpen(false);
        setIsSubcategoryModalOpen(false);
      } catch (error) {
        console.error('Errore eliminazione categoria:', error);
      } finally {
        setCategoryToDelete(null);
      }
    }
  };

  const handleCategorySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (isEditMode && selectedCategory) {
        const updated = await apiService.updateCategory(selectedCategory.id, categoryFormData);
        updateCategoryCache(updated);
      } else {
        const newCategory = await apiService.createCategory(categoryFormData);
        addCategory(newCategory);
      }
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
      const newSubcategory = await apiService.createSubcategory(selectedCategory.id, subcategoryFormData);

      // Aggiorna la categoria nella cache con la nuova sottocategoria
      const categoryFromCache = baseCategories.find(c => c.id === selectedCategory.id);
      if (categoryFromCache) {
        const updatedCategory = {
          ...categoryFromCache,
          subcategories: [...(categoryFromCache.subcategories || []), newSubcategory]
        };
        updateCategoryCache(updatedCategory);
      }

      setSubcategoryFormData({ name: '' });
      setShowSubcategoryForm(false);
      // Aggiorna la categoria selezionata con le nuove statistiche
      const updated = categories.find(c => c.id === selectedCategory.id);
      if (updated) setSelectedCategory(updated);
    } catch (error) {
      console.error('Errore creazione sottocategoria:', error);
    }
  };

  const handleDeleteSubcategory = (subcategoryId: number) => {
    setSubcategoryToDelete(subcategoryId);
    setIsDeleteSubcategoryDialogOpen(true);
  };

  const confirmDeleteSubcategory = async () => {
    if (!selectedCategory || subcategoryToDelete === null) return;
    try {
      await apiService.deleteSubcategory(selectedCategory.id, subcategoryToDelete);

      // Aggiorna la categoria nella cache rimuovendo la sottocategoria
      const categoryFromCache = baseCategories.find(c => c.id === selectedCategory.id);
      if (categoryFromCache) {
        const updatedCategory = {
          ...categoryFromCache,
          subcategories: (categoryFromCache.subcategories || []).filter(s => s.id !== subcategoryToDelete)
        };
        updateCategoryCache(updatedCategory);
      }

      // Aggiorna la categoria selezionata con le nuove statistiche
      const updated = categories.find(c => c.id === selectedCategory.id);
      if (updated) setSelectedCategory(updated);
    } catch (error) {
      console.error('Errore eliminazione sottocategoria:', error);
    } finally {
      setSubcategoryToDelete(null);
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
          onClose={() => {
            setIsSubcategoryModalOpen(false);
            setShowSubcategoryForm(false);
            setSubcategoryFormData({ name: '' });
          }}
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
                &nbsp;&nbsp;⚙️
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
                      <span className="text-sm text-gray-900 dark:text-gray-100">{sub.name}</span>
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
            {!showSubcategoryForm ? (
              <div className="flex justify-center py-3">
                <button
                  type="button"
                  onClick={() => setShowSubcategoryForm(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors text-sm font-medium"
                >
                  <span className="text-xl">+</span>
                  <span>Nuova Sottocategoria</span>
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubcategorySubmit} className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Nuova Sottocategoria
                </h4>

                <div className="flex gap-2">
                  <input
                    type="text"
                    value={subcategoryFormData.name}
                    onChange={(e) => setSubcategoryFormData({ ...subcategoryFormData, name: e.target.value })}
                    className="flex-[3] px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
                    placeholder="Nome sottocategoria"
                    required
                    autoFocus
                  />
                  <button
                    type="submit"
                    className="flex-1 h-10 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors flex items-center justify-center text-2xl"
                    title="Aggiungi"
                  >
                    ✓
                  </button>
                </div>
              </form>
            )}
          </div>
        </Modal>

        {/* Modal categoria */}
        <Modal
          isOpen={isCategoryModalOpen}
          onClose={() => {
            setIsCategoryModalOpen(false);
          }}
          onBackdropClick={() => {
            setIsCategoryModalOpen(false);
            setIsSubcategoryModalOpen(false);
            setShowSubcategoryForm(false);
            setSubcategoryFormData({ name: '' });
          }}
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
              <div className="grid grid-cols-6 gap-2 max-h-64 overflow-y-auto">
                {getSuggestedIcons(categoryFormData.name).map((icon) => (
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
              {isEditMode && selectedCategory ? (
                <>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteCategory(e, selectedCategory.id)}
                    className="flex-1 h-12 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors flex items-center justify-center text-2xl"
                    title="Elimina"
                  >
                    🗑️
                  </button>
                  <button
                    type="submit"
                    className="flex-1 h-12 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors flex items-center justify-center text-2xl"
                    title="Salva"
                  >
                    ✓
                  </button>
                </>
              ) : (
                <button
                  type="submit"
                  className="w-full h-12 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-colors flex items-center justify-center text-2xl"
                  title="Crea"
                >
                  ✓
                </button>
              )}
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

        {/* Confirm Dialogs */}
        <ConfirmDialog
          isOpen={isDeleteCategoryDialogOpen}
          onClose={() => setIsDeleteCategoryDialogOpen(false)}
          onConfirm={confirmDeleteCategory}
          title="Elimina Categoria"
          message="Sei sicuro di voler eliminare questa categoria? Verranno eliminate anche tutte le sottocategorie associate."
          confirmText="Elimina"
          cancelText="Annulla"
          isDestructive={true}
        />

        <ConfirmDialog
          isOpen={isDeleteSubcategoryDialogOpen}
          onClose={() => setIsDeleteSubcategoryDialogOpen(false)}
          onConfirm={confirmDeleteSubcategory}
          title="Elimina Sottocategoria"
          message="Sei sicuro di voler eliminare questa sottocategoria?"
          confirmText="Elimina"
          cancelText="Annulla"
          isDestructive={true}
        />
      </div>
    </Layout>
  );
}
