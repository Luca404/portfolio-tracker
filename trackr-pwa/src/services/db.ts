import type {
  Transaction,
  Category,
  Subcategory,
  Account,
  Portfolio,
  User,
} from '../types';

const DB_NAME = 'trackr-db';
const DB_VERSION = 1;

// Definizione degli store
export const STORES = {
  USERS: 'users',
  TRANSACTIONS: 'transactions',
  CATEGORIES: 'categories',
  SUBCATEGORIES: 'subcategories',
  ACCOUNTS: 'accounts',
  PORTFOLIOS: 'portfolios',
} as const;

class DatabaseService {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<IDBDatabase> | null = null;

  async init(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        reject(new Error('Errore apertura database'));
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Store utenti
        if (!db.objectStoreNames.contains(STORES.USERS)) {
          const userStore = db.createObjectStore(STORES.USERS, { keyPath: 'id' });
          userStore.createIndex('username', 'username', { unique: true });
        }

        // Store transazioni
        if (!db.objectStoreNames.contains(STORES.TRANSACTIONS)) {
          const txStore = db.createObjectStore(STORES.TRANSACTIONS, { keyPath: 'id', autoIncrement: true });
          txStore.createIndex('userId', 'userId', { unique: false });
          txStore.createIndex('date', 'date', { unique: false });
          txStore.createIndex('type', 'type', { unique: false });
          txStore.createIndex('category', 'category', { unique: false });
          txStore.createIndex('account_id', 'account_id', { unique: false });
        }

        // Store categorie
        if (!db.objectStoreNames.contains(STORES.CATEGORIES)) {
          const catStore = db.createObjectStore(STORES.CATEGORIES, { keyPath: 'id', autoIncrement: true });
          catStore.createIndex('userId', 'user_id', { unique: false });
          catStore.createIndex('name', 'name', { unique: false });
        }

        // Store sottocategorie
        if (!db.objectStoreNames.contains(STORES.SUBCATEGORIES)) {
          const subStore = db.createObjectStore(STORES.SUBCATEGORIES, { keyPath: 'id', autoIncrement: true });
          subStore.createIndex('categoryId', 'category_id', { unique: false });
        }

        // Store account
        if (!db.objectStoreNames.contains(STORES.ACCOUNTS)) {
          const accStore = db.createObjectStore(STORES.ACCOUNTS, { keyPath: 'id', autoIncrement: true });
          accStore.createIndex('userId', 'user_id', { unique: false });
        }

        // Store portfolio
        if (!db.objectStoreNames.contains(STORES.PORTFOLIOS)) {
          const portStore = db.createObjectStore(STORES.PORTFOLIOS, { keyPath: 'id', autoIncrement: true });
          portStore.createIndex('userId', 'user_id', { unique: false });
        }
      };
    });

    return this.initPromise;
  }

  // ==================== UTILITY GENERICHE ====================

  async add<T extends { id?: any; created_at?: string; updated_at?: string }>(
    storeName: string,
    data: Omit<T, 'id' | 'created_at' | 'updated_at'>
  ): Promise<T> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);

      const dataWithTimestamp = {
        ...data,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      const request = store.add(dataWithTimestamp);

      request.onsuccess = () => {
        resolve({ ...dataWithTimestamp, id: request.result } as T);
      };

      request.onerror = () => {
        reject(new Error(`Errore aggiunta in ${storeName}`));
      };
    });
  }

  async getAll<T>(storeName: string, indexName?: string, indexValue?: IDBValidKey): Promise<T[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);

      let request: IDBRequest;

      if (indexName && indexValue !== undefined) {
        const index = store.index(indexName);
        request = index.getAll(indexValue);
      } else {
        request = store.getAll();
      }

      request.onsuccess = () => {
        resolve(request.result as T[]);
      };

      request.onerror = () => {
        reject(new Error(`Errore lettura da ${storeName}`));
      };
    });
  }

  async getById<T>(storeName: string, id: string | number): Promise<T | null> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);

      request.onsuccess = () => {
        resolve(request.result || null);
      };

      request.onerror = () => {
        reject(new Error(`Errore lettura da ${storeName}`));
      };
    });
  }

  async update<T>(storeName: string, id: string | number, data: Partial<T>): Promise<T> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const getRequest = store.get(id);

      getRequest.onsuccess = () => {
        const existing = getRequest.result;
        if (!existing) {
          reject(new Error(`Elemento non trovato in ${storeName}`));
          return;
        }

        const updated = {
          ...existing,
          ...data,
          id,
          updated_at: new Date().toISOString(),
        };

        const putRequest = store.put(updated);

        putRequest.onsuccess = () => {
          resolve(updated as T);
        };

        putRequest.onerror = () => {
          reject(new Error(`Errore aggiornamento in ${storeName}`));
        };
      };

      getRequest.onerror = () => {
        reject(new Error(`Errore lettura da ${storeName}`));
      };
    });
  }

  async delete(storeName: string, id: string | number): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(id);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Errore eliminazione da ${storeName}`));
      };
    });
  }

  async clear(storeName: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Errore pulizia ${storeName}`));
      };
    });
  }

  // ==================== METODI SPECIFICI ====================

  // Transazioni con filtri (single-user: ritorna tutte senza filtro userId)
  async getTransactions(filters?: {
    startDate?: string;
    endDate?: string;
    category?: string;
    type?: string;
  }): Promise<Transaction[]> {
    let transactions = await this.getAll<Transaction>(STORES.TRANSACTIONS);

    if (filters?.startDate) {
      transactions = transactions.filter(tx => tx.date >= filters.startDate!);
    }
    if (filters?.endDate) {
      transactions = transactions.filter(tx => tx.date <= filters.endDate!);
    }
    if (filters?.category) {
      transactions = transactions.filter(tx => tx.category === filters.category);
    }
    if (filters?.type) {
      transactions = transactions.filter(tx => tx.type === filters.type);
    }

    return transactions.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }

  // Categorie con sottocategorie (single-user: ritorna tutte senza filtro userId)
  async getCategoriesWithSubcategories(): Promise<Category[]> {
    const categories = await this.getAll<Category>(STORES.CATEGORIES);

    const categoriesWithSubs = await Promise.all(
      categories.map(async (cat) => {
        const subcategories = await this.getAll<Subcategory>(
          STORES.SUBCATEGORIES,
          'categoryId',
          cat.id
        );
        return { ...cat, subcategories };
      })
    );

    return categoriesWithSubs;
  }

  // Elimina categoria e sottocategorie
  async deleteCategory(id: number): Promise<void> {
    const subcategories = await this.getAll<Subcategory>(STORES.SUBCATEGORIES, 'categoryId', id);

    // Elimina tutte le sottocategorie
    await Promise.all(
      subcategories.map(sub => this.delete(STORES.SUBCATEGORIES, sub.id))
    );

    // Elimina la categoria
    await this.delete(STORES.CATEGORIES, id);
  }

  // ==================== EXPORT/IMPORT ====================

  async exportData(): Promise<string> {
    const userId = this.getCurrentUserId();

    const [transactions, categories, accounts, portfolios] = await Promise.all([
      this.getAll<Transaction>(STORES.TRANSACTIONS, 'userId', userId),
      this.getCategoriesWithSubcategories(),
      this.getAll<Account>(STORES.ACCOUNTS, 'userId', userId),
      this.getAll<Portfolio>(STORES.PORTFOLIOS, 'userId', userId),
    ]);

    const exportData = {
      version: DB_VERSION,
      exportDate: new Date().toISOString(),
      userId,
      data: {
        transactions,
        categories,
        accounts,
        portfolios,
      },
    };

    return JSON.stringify(exportData, null, 2);
  }

  async importData(jsonData: string): Promise<void> {
    const importData = JSON.parse(jsonData);
    const userId = this.getCurrentUserId();

    // Verifica versione compatibile
    if (importData.version > DB_VERSION) {
      throw new Error('Versione file di backup non compatibile');
    }

    // Pulisci dati esistenti dell'utente corrente
    await this.clearUserData(userId);

    // Importa nuovi dati
    const { transactions, categories, accounts, portfolios } = importData.data;

    // Importa categorie e sottocategorie
    for (const category of categories) {
      const { subcategories, id, ...categoryData } = category;
      const newCategory = await this.add<Category>(STORES.CATEGORIES, {
        ...categoryData,
        user_id: userId,
      });

      // Importa sottocategorie
      for (const sub of subcategories || []) {
        const { id: subId, ...subData } = sub;
        await this.add<Subcategory>(STORES.SUBCATEGORIES, {
          ...subData,
          category_id: newCategory.id,
        });
      }
    }

    // Importa accounts
    for (const account of accounts || []) {
      const { id, ...accountData } = account;
      await this.add<Account>(STORES.ACCOUNTS, {
        ...accountData,
        user_id: userId,
      });
    }

    // Importa portfolios
    for (const portfolio of portfolios || []) {
      const { id, ...portfolioData } = portfolio;
      await this.add<Portfolio>(STORES.PORTFOLIOS, {
        ...portfolioData,
        user_id: userId,
      });
    }

    // Importa transazioni
    for (const transaction of transactions || []) {
      const { id, ...txData } = transaction;
      await this.add<Transaction>(STORES.TRANSACTIONS, {
        ...txData,
        userId,
      });
    }
  }

  async clearUserData(userId: string): Promise<void> {
    const [transactions, categories, accounts, portfolios] = await Promise.all([
      this.getAll<Transaction>(STORES.TRANSACTIONS, 'userId', userId),
      this.getAll<Category>(STORES.CATEGORIES, 'userId', userId),
      this.getAll<Account>(STORES.ACCOUNTS, 'userId', userId),
      this.getAll<Portfolio>(STORES.PORTFOLIOS, 'userId', userId),
    ]);

    // Elimina tutte le transazioni
    await Promise.all(transactions.map(tx => this.delete(STORES.TRANSACTIONS, tx.id)));

    // Elimina categorie (e sottocategorie)
    await Promise.all(categories.map(cat => this.deleteCategory(cat.id)));

    // Elimina accounts
    await Promise.all(accounts.map(acc => this.delete(STORES.ACCOUNTS, acc.id)));

    // Elimina portfolios
    await Promise.all(portfolios.map(port => this.delete(STORES.PORTFOLIOS, port.id)));
  }

  // ==================== AUTENTICAZIONE ====================

  getCurrentUserId(): string {
    return 'local-user';
  }

  async createOrUpdateUser(user: User): Promise<User> {
    const existing = await this.getById<User>(STORES.USERS, user.id);

    if (existing) {
      return this.update<User>(STORES.USERS, user.id, user);
    } else {
      return this.add<User>(STORES.USERS, user);
    }
  }
}

export const dbService = new DatabaseService();
