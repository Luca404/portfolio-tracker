# Setup Completo Trackr PWA + Backend

Guida completa per avviare sia il backend pfTrackr aggiornato che la PWA Trackr.

## Architettura

```
portfolio-tracker/
├── backend/              # FastAPI backend (Python)
│   ├── main.py          # ✅ Aggiornato con transactions_router
│   ├── models/          # ✅ Aggiunto TransactionModel
│   ├── schemas/         # ✅ Aggiunto TransactionCreate/Response/Stats
│   └── routers/         # ✅ Aggiunto transactions.py
├── frontend/            # React frontend pfTrackr (esistente)
└── trackr-pwa/          # 🆕 PWA gestione spese (nuovo)
```

## 1. Setup Backend (Porta 8000)

### Avvio Backend

Il backend è già configurato per gestire sia pfTrackr che Trackr.

```bash
cd backend

# Assicurati che il virtual environment sia attivo
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows

# Avvia il server
python main.py
# oppure
uvicorn main:app --reload --port 8000
```

Il backend sarà disponibile su `http://localhost:8000`

**CONFIGURAZIONE IMPORTANTE**:
- Il backend DEVE girare sulla porta **8000**
- Il frontend usa percorsi relativi `/api/*` che vengono inoltrati dal proxy Vite
- Non serve cambiare nessuna configurazione

### Nuovi Endpoints Aggiunti

Il backend ora include questi nuovi endpoints per Trackr:

**Auth (modificati per compatibilità):**
- `POST /api/auth/register` - Registrazione (ora accetta `name` invece di `username`)
- `POST /api/auth/login` - Login (risposta modificata con campo `token`)

**Transactions (nuovi):**
- `POST /api/transactions` - Crea transazione
- `GET /api/transactions` - Lista transazioni (supporta filtri: startDate, endDate, category, type)
- `GET /api/transactions/stats` - Statistiche aggregate
- `GET /api/transactions/{id}` - Dettaglio transazione
- `PUT /api/transactions/{id}` - Aggiorna transazione
- `DELETE /api/transactions/{id}` - Elimina transazione

### Migrazione Database

Le nuove tabelle verranno create automaticamente all'avvio:

```python
# In backend/main.py (già presente)
Base.metadata.create_all(bind=db_engine)
```

Verrà creata la tabella `transactions` con questi campi:
- id, user_id, type, category, amount, description, date
- ticker, quantity, price (per investimenti)
- created_at, updated_at

## 2. Setup Trackr PWA (Porta 5174)

### Installazione e Avvio

```bash
cd trackr-pwa

# Installa dipendenze (se non l'hai già fatto)
npm install

# Avvia dev server
npm run dev
```

L'app sarà disponibile su `http://localhost:5174`

### Primo Utilizzo

1. Apri `http://localhost:5174`
2. Clicca su "Registrati"
3. Inserisci:
   - Nome: `Mario Rossi`
   - Email: `mario@test.com`
   - Password: `password123`
4. Verrà creato un account e sarai automaticamente loggato

### Test Integrazione Investimenti

1. Nella dashboard, clicca sul FAB (+) in basso a destra
2. Seleziona tipo "Uscita"
3. Scegli categoria "Investimento"
4. Compila:
   - Importo: `1000`
   - Ticker: `AAPL`
   - Quantità: `5`
   - Prezzo: `200`
   - Data: oggi
   - Descrizione: `Acquisto azioni Apple`
5. Clicca "Salva"

**Cosa succede:**
- La transazione viene salvata in Trackr
- Viene creato automaticamente un Order in pfTrackr
- Se apri pfTrackr (frontend), vedrai l'ordine nel portfolio

## 3. Verifica Funzionamento

### Test Backend

Verifica che gli endpoints funzionino:

```bash
# Registrazione
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","name":"Test User","password":"test123"}'

# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123"}'

# Lista transazioni (richiede token JWT)
curl http://localhost:5000/api/transactions \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### Test Frontend Trackr PWA

1. **Login/Register**: Funziona? Vedi il token in localStorage?
2. **Dashboard**: Vedi le card con statistiche?
3. **Form Transazione**: Il modal si apre? I campi investimento appaiono?
4. **Lista Transazioni**: Vedi le transazioni create?
5. **PWA**: Clicca "Install" nella barra del browser

### Debug Comune

**Errore CORS:**
```
Access to fetch at 'http://localhost:5000' from origin 'http://localhost:5174' has been blocked
```
Soluzione: Il backend ha già CORS configurato, riavvia il server backend.

**401 Unauthorized:**
Il token JWT è scaduto o non valido. Fai logout e login di nuovo.

**Transazioni non appaiono:**
Controlla che il filtro data sia corretto (default: mese corrente).

## 4. Build per Produzione

### Backend

```bash
cd backend
# Il backend è già pronto per produzione
# Usa gunicorn o uvicorn con workers
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4
```

### Trackr PWA

```bash
cd trackr-pwa

# Build
npm run build

# I file saranno in dist/
# Servili con un server statico (nginx, vercel, netlify)

# Test build locale
npm run preview
```

## 5. Sviluppi Futuri

Possibili miglioramenti:

- [ ] Grafici interattivi con Chart.js/Recharts
- [ ] Export dati in CSV/PDF
- [ ] Notifiche push per promemoria spese
- [ ] Sincronizzazione bidirezionale con pfTrackr
- [ ] Categorie personalizzabili
- [ ] Budget mensili con alert
- [ ] Ricorrenze automatiche (abbonamenti)
- [ ] Multi-valuta

## 6. Troubleshooting

### Backend non parte

```bash
# Verifica dipendenze Python
pip install -r requirements.txt

# Verifica porta occupata
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows
```

### Frontend non compila

```bash
# Pulisci cache
rm -rf node_modules package-lock.json
npm install

# Verifica versione Node (deve essere >= 18)
node --version
```

### Database errors

```bash
# Elimina e ricrea database (ATTENZIONE: perdi tutti i dati!)
rm backend/database.db
# Riavvia backend, creerà nuovo database
```

## Contatti

Per problemi o domande, apri una issue su GitHub.

---

**Buon tracking delle spese! 💰📊**
