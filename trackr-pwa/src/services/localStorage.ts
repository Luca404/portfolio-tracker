import { dbService, STORES } from './db';
import type {
  AuthResponse,
  LoginCredentials,
  RegisterCredentials,
  Transaction,
  TransactionFormData,
  TransactionStats,
  User,
  Category,
  CategoryWithStats,
  CategoryFormData,
  Subcategory,
  SubcategoryFormData,
  Account,
  AccountFormData,
  Portfolio,
  PortfolioFormData,
  SubcategoryWithStats,
} from '../types';

class LocalStorageService {
  // ==================== AUTH ====================

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    // Inizializza il database
    await dbService.init();

    // In modalità locale, crea un utente se non esiste
    const user: User = {
      id: 'local-user',
      name: credentials.username,
      createdAt: new Date().toISOString(),
    };

    await dbService.createOrUpdateUser(user);

    // Simula un token
    const authResponse: AuthResponse = {
      access_token: 'local-token',
      token_type: 'bearer',
      user,
    };

    // Salva in localStorage
    localStorage.setItem('access_token', authResponse.access_token);
    localStorage.setItem('authToken', authResponse.access_token);
    localStorage.setItem('user', JSON.stringify(user));

    return authResponse;
  }

  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    // In modalità locale, register = login
    return this.login(credentials);
  }

  logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }

  getCurrentUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('authToken');
  }

  // ==================== TRANSACTIONS ====================

  async getTransactions(params?: {
    startDate?: string;
    endDate?: string;
    category?: string;
    type?: string;
  }): Promise<Transaction[]> {
    return dbService.getTransactions({
      startDate: params?.startDate,
      endDate: params?.endDate,
      category: params?.category,
      type: params?.type,
    });
  }

  async getTransaction(id: string): Promise<Transaction> {
    const transaction = await dbService.getById<Transaction>(STORES.TRANSACTIONS, id);
    if (!transaction) {
      throw new Error('Transazione non trovata');
    }
    return transaction;
  }

  async createTransaction(data: TransactionFormData): Promise<Transaction> {
    const userId = dbService.getCurrentUserId();

    const transactionData: Omit<Transaction, 'id'> = {
      ...data,
      userId: String(userId),
      account_id: data.account_id || 0,
    };

    return dbService.add<Transaction>(STORES.TRANSACTIONS, transactionData);
  }

  async updateTransaction(id: string, data: Partial<TransactionFormData>): Promise<Transaction> {
    return dbService.update<Transaction>(STORES.TRANSACTIONS, id, data);
  }

  async deleteTransaction(id: string): Promise<void> {
    return dbService.delete(STORES.TRANSACTIONS, id);
  }

  async getTransactionStats(params?: {
    startDate?: string;
    endDate?: string;
  }): Promise<TransactionStats> {
    const transactions = await this.getTransactions({
      startDate: params?.startDate,
      endDate: params?.endDate,
    });

    const stats: TransactionStats = {
      totalExpenses: 0,
      totalIncome: 0,
      totalInvestments: 0,
      balance: 0,
      expensesByCategory: {},
      monthlyTrend: [],
    };

    // Calcola statistiche
    transactions.forEach((tx) => {
      if (tx.type === 'expense') {
        stats.totalExpenses += tx.amount;
        stats.expensesByCategory[tx.category] =
          (stats.expensesByCategory[tx.category] || 0) + tx.amount;
      } else if (tx.type === 'income') {
        stats.totalIncome += tx.amount;
      } else if (tx.type === 'investment') {
        stats.totalInvestments += tx.amount;
      }
    });

    stats.balance = stats.totalIncome - stats.totalExpenses - stats.totalInvestments;

    // Calcola trend mensile
    const monthlyData: Record<string, { expenses: number; income: number }> = {};

    transactions.forEach((tx) => {
      const month = tx.date.substring(0, 7); // YYYY-MM
      if (!monthlyData[month]) {
        monthlyData[month] = { expenses: 0, income: 0 };
      }

      if (tx.type === 'expense') {
        monthlyData[month].expenses += tx.amount;
      } else if (tx.type === 'income') {
        monthlyData[month].income += tx.amount;
      }
    });

    stats.monthlyTrend = Object.entries(monthlyData)
      .map(([month, data]) => ({ month, ...data }))
      .sort((a, b) => a.month.localeCompare(b.month));

    return stats;
  }

  // ==================== CATEGORIES ====================

  // Mutex per evitare race condition nella creazione dei default
  private _defaultCategoriesPromise: Promise<void> | null = null;
  private _defaultAccountsPromise: Promise<void> | null = null;

  // Categorie predefinite (come nel backend)
  private DEFAULT_CATEGORIES = [
    { name: 'Alimentari', icon: '🍔', category_type: 'expense' },
    { name: 'Trasporti', icon: '🚗', category_type: 'expense' },
    { name: 'Utenze', icon: '⚡', category_type: 'expense' },
    { name: 'Svago', icon: '🎮', category_type: 'expense' },
    { name: 'Salute', icon: '🏥', category_type: 'expense' },
    { name: 'Shopping', icon: '🛍️', category_type: 'expense' },
    { name: 'Investimento', icon: '💰', category_type: 'investment' },
    { name: 'Stipendio', icon: '💵', category_type: 'income' },
    { name: 'Bonus', icon: '🎁', category_type: 'income' },
    { name: 'Trasferimento', icon: '🔄', category_type: 'transfer' },
    { name: 'Altro', icon: '📌', category_type: null as any }, // Categoria generica per tutti i tipi
  ];

  private createDefaultCategories(): Promise<void> {
    // Se c'è già una creazione in corso, ritorna la stessa Promise (evita race condition)
    if (this._defaultCategoriesPromise) return this._defaultCategoriesPromise;

    this._defaultCategoriesPromise = (async () => {
      const allCategories = await dbService.getAll<Category>(STORES.CATEGORIES);
      const userId = dbService.getCurrentUserId();

      // Controlla quali gruppi sono vuoti (separati per tab UI)
      // Uscite: expense + null (Altro)
      const hasExpense = allCategories.some(c => c.category_type === 'expense' || c.category_type == null);
      // Entrate: income
      const hasIncome = allCategories.some(c => c.category_type === 'income');
      // Investimenti: investment
      const hasInvestment = allCategories.some(c => c.category_type === 'investment');

      // Crea defaults solo per i gruppi vuoti
      for (const catData of this.DEFAULT_CATEGORIES) {
        const type = catData.category_type;
        const isExpenseGroup = type === 'expense' || type == null;
        const isIncomeGroup = type === 'income';
        const isInvestmentGroup = type === 'investment';

        const shouldCreate =
          (isExpenseGroup && !hasExpense) ||
          (isIncomeGroup && !hasIncome) ||
          (isInvestmentGroup && !hasInvestment);

        if (shouldCreate) {
          await dbService.add<Category>(STORES.CATEGORIES, {
            name: catData.name,
            icon: catData.icon,
            category_type: catData.category_type,
            user_id: userId as any,
            subcategories: [],
          });
        }
      }
    })().finally(() => {
      this._defaultCategoriesPromise = null;
    });

    return this._defaultCategoriesPromise;
  }

  async getCategories(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<CategoryWithStats[]> {
    // Crea categorie di default se necessario
    await this.createDefaultCategories();

    const categories = await dbService.getCategoriesWithSubcategories();

    // Se ci sono parametri di date, calcola statistiche
    if (params?.start_date || params?.end_date) {
      const transactions = await this.getTransactions({
        startDate: params.start_date,
        endDate: params.end_date,
      });

      return categories.map((cat) => {
        const categoryTxs = transactions.filter((tx) => tx.category === cat.name);
        const total_amount = categoryTxs.reduce((sum, tx) => sum + tx.amount, 0);

        const subcategoriesWithStats: SubcategoryWithStats[] = cat.subcategories.map((sub) => {
          const subTxs = categoryTxs.filter((tx) => tx.subcategory === sub.name);
          return {
            ...sub,
            total_amount: subTxs.reduce((sum, tx) => sum + tx.amount, 0),
            transaction_count: subTxs.length,
          };
        });

        return {
          ...cat,
          total_amount,
          transaction_count: categoryTxs.length,
          subcategories: subcategoriesWithStats,
        };
      });
    }

    // Senza parametri, ritorna categorie senza stats
    return categories.map((cat) => ({
      ...cat,
      total_amount: 0,
      transaction_count: 0,
      subcategories: cat.subcategories.map((sub) => ({
        ...sub,
        total_amount: 0,
        transaction_count: 0,
      })),
    }));
  }

  async createCategory(data: CategoryFormData): Promise<Category> {
    const userId = dbService.getCurrentUserId();

    // Genera un ID numerico
    const categories = await dbService.getAll<Category>(STORES.CATEGORIES);
    const maxId = categories.reduce((max, cat) => Math.max(max, cat.id), 0);

    const categoryData = {
      ...data,
      user_id: userId,
      subcategories: [],
    };

    return dbService.add<Category>(STORES.CATEGORIES, categoryData);
  }

  async updateCategory(id: number, data: Partial<CategoryFormData>): Promise<Category> {
    return dbService.update<Category>(STORES.CATEGORIES, id, data);
  }

  async deleteCategory(id: number): Promise<void> {
    return dbService.deleteCategory(id);
  }

  // ==================== SUBCATEGORIES ====================

  async getSubcategories(categoryId: number): Promise<Subcategory[]> {
    return dbService.getAll<Subcategory>(STORES.SUBCATEGORIES, 'categoryId', categoryId);
  }

  async createSubcategory(categoryId: number, data: SubcategoryFormData): Promise<Subcategory> {
    const subcategoryData = {
      ...data,
      category_id: categoryId,
    };

    return dbService.add<Subcategory>(STORES.SUBCATEGORIES, subcategoryData);
  }

  async updateSubcategory(
    categoryId: number,
    subcategoryId: number,
    data: Partial<SubcategoryFormData>
  ): Promise<Subcategory> {
    return dbService.update<Subcategory>(STORES.SUBCATEGORIES, subcategoryId, data);
  }

  async deleteSubcategory(categoryId: number, subcategoryId: number): Promise<void> {
    return dbService.delete(STORES.SUBCATEGORIES, subcategoryId);
  }

  // ==================== ACCOUNTS ====================

  private createDefaultAccounts(): Promise<void> {
    // Se c'è già una creazione in corso, ritorna la stessa Promise (evita race condition)
    if (this._defaultAccountsPromise) return this._defaultAccountsPromise;

    this._defaultAccountsPromise = (async () => {
      const allAccounts = await dbService.getAll<Account>(STORES.ACCOUNTS);
      if (allAccounts.length > 0) return;

      const userId = dbService.getCurrentUserId();
      await dbService.add<Account>(STORES.ACCOUNTS, {
        user_id: userId as any,
        name: 'Conto Corrente',
        icon: '🏦',
        initial_balance: 0,
        is_favorite: true,
        current_balance: 0,
      });

      await dbService.add<Account>(STORES.ACCOUNTS, {
        user_id: userId as any,
        name: 'Contanti',
        icon: '💵',
        initial_balance: 0,
        is_favorite: false,
        current_balance: 0,
      });
    })().finally(() => {
      this._defaultAccountsPromise = null;
    });

    return this._defaultAccountsPromise;
  }

  async getAccounts(): Promise<Account[]> {
    // Crea account di default se necessario
    await this.createDefaultAccounts();

    // Legge tutti gli account (senza filtro userId per retrocompatibilità)
    const accounts = await dbService.getAll<Account>(STORES.ACCOUNTS);

    // Calcola current_balance per ogni account
    const transactions = await this.getTransactions();

    return accounts.map((account) => {
      const accountTxs = transactions.filter((tx) => tx.account_id === account.id);

      const balance = accountTxs.reduce((sum, tx) => {
        if (tx.type === 'income') return sum + tx.amount;
        if (tx.type === 'expense') return sum - tx.amount;
        if (tx.type === 'investment') return sum - tx.amount;
        return sum;
      }, account.initial_balance);

      return {
        ...account,
        current_balance: balance,
      };
    });
  }

  async createAccount(data: AccountFormData): Promise<Account> {
    const userId = dbService.getCurrentUserId();

    const accountData = {
      ...data,
      user_id: userId,
      current_balance: data.initial_balance,
    };

    return dbService.add<Account>(STORES.ACCOUNTS, accountData);
  }

  async updateAccount(id: number, data: Partial<AccountFormData>): Promise<Account> {
    return dbService.update<Account>(STORES.ACCOUNTS, id, data);
  }

  async deleteAccount(id: number): Promise<void> {
    return dbService.delete(STORES.ACCOUNTS, id);
  }

  // ==================== PORTFOLIOS ====================

  async getPortfolios(): Promise<Portfolio[]> {
    const userId = dbService.getCurrentUserId();
    return dbService.getAll<Portfolio>(STORES.PORTFOLIOS, 'userId', userId);
  }

  async createPortfolio(data: PortfolioFormData): Promise<Portfolio> {
    const userId = dbService.getCurrentUserId();

    const portfolioData = {
      ...data,
      user_id: userId,
      initial_capital: data.initial_capital || 0,
      reference_currency: data.reference_currency || 'EUR',
      risk_free_source: data.risk_free_source || '',
      market_benchmark: data.market_benchmark || '',
    };

    return dbService.add<Portfolio>(STORES.PORTFOLIOS, portfolioData);
  }

  async updatePortfolio(id: number, data: Partial<PortfolioFormData>): Promise<Portfolio> {
    return dbService.update<Portfolio>(STORES.PORTFOLIOS, id, data);
  }

  async deletePortfolio(id: number): Promise<void> {
    return dbService.delete(STORES.PORTFOLIOS, id);
  }

  // ==================== EXPORT/IMPORT ====================

  async exportData(): Promise<void> {
    const jsonData = await dbService.exportData();
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trackr-backup-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async importData(file: File): Promise<void> {
    const text = await file.text();
    await dbService.importData(text);
  }
}

export const localStorageService = new LocalStorageService();
