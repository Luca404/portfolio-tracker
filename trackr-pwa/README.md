# Trackr PWA - Gestione Spese Personali

Progressive Web App mobile-first per tracciare spese quotidiane stile Kakebo, con integrazione al backend pfTrackr per sincronizzare investimenti.

## Caratteristiche

- **PWA Installabile**: Installabile su dispositivi mobili e desktop
- **Offline-Ready**: Funziona offline grazie a Service Worker e cache
- **Mobile-First**: UI ottimizzata per dispositivi touch
- **Dark Mode**: Supporto per tema scuro
- **Autenticazione JWT**: Login sicuro con token
- **Integrazione pfTrackr**: Investimenti sincronizzati automaticamente con il portfolio tracker

## Tecnologie

- React 18 + TypeScript
- Vite
- Tailwind CSS
- React Router
- Axios
- Vite PWA Plugin
- Workbox

## Setup

### 1. Installazione dipendenze

```bash
cd trackr-pwa
npm install
```

### 2. Configurazione ambiente

Copia il file `.env.example` in `.env` e configura l'URL del backend:

```bash
cp .env.example .env
```

Modifica `.env`:
```
VITE_API_URL=http://localhost:5000
```

### 3. Avvio server di sviluppo

```bash
npm run dev
```

L'app sarà disponibile su [http://localhost:5174](http://localhost:5174)

### 4. Build per produzione

```bash
npm run build
```

I file compilati saranno nella cartella `dist/`.

## Struttura Progetto

```
trackr-pwa/
├── src/
│   ├── components/       # Componenti React
│   │   ├── common/       # Componenti riutilizzabili (FAB, Modal, LoadingSpinner)
│   │   ├── layout/       # Layout e navigazione
│   │   └── transactions/ # Componenti specifici per transazioni
│   ├── contexts/         # React Contexts (Auth)
│   ├── pages/           # Pagine dell'app
│   │   ├── LoginPage.tsx
│   │   ├── DashboardPage.tsx
│   │   ├── TransactionsPage.tsx
│   │   └── StatsPage.tsx
│   ├── services/        # Servizi API
│   │   └── api.ts       # Client API con interceptors JWT
│   ├── types/           # TypeScript interfaces
│   │   └── index.ts
│   ├── App.tsx          # Root component con routing
│   ├── main.tsx         # Entry point
│   └── index.css        # Stili Tailwind
├── public/              # Asset statici
├── vite.config.ts       # Configurazione Vite + PWA
├── tailwind.config.js   # Configurazione Tailwind
└── package.json
```

## Funzionalità

### Autenticazione

- Registrazione nuovi utenti
- Login con email/password
- Token JWT salvato in localStorage
- Logout

### Dashboard

- Overview mensile: entrate, uscite, investimenti, bilancio
- Spese per categoria (top 5)
- Ultimi 5 transazioni
- FAB per aggiungere velocemente una spesa

### Transazioni

- Lista completa di tutte le transazioni
- Filtri per tipo (entrate/uscite/investimenti)
- Eliminazione transazioni
- Dettagli investimenti (ticker, quantità, prezzo)

### Statistiche

- Vista mensile/annuale
- Distribuzione per categoria con grafici a barre
- Trend mensile
- Totale investimenti sincronizzati con pfTrackr

### Form Inserimento

- Tipo: Entrata/Uscita
- Categoria selezionabile
- Importo con tastierino numerico ottimizzato
- Data
- Descrizione opzionale
- **Campi speciali per Investimenti**:
  - Ticker/Simbolo
  - Quantità
  - Prezzo
  - Sincronizzazione automatica con pfTrackr

## Integrazione Backend

### Endpoints Utilizzati

**Auth:**
- `POST /api/auth/register` - Registrazione
- `POST /api/auth/login` - Login

**Transactions:**
- `GET /api/transactions` - Lista transazioni (con filtri)
- `POST /api/transactions` - Crea transazione
- `GET /api/transactions/:id` - Dettaglio transazione
- `PUT /api/transactions/:id` - Aggiorna transazione
- `DELETE /api/transactions/:id` - Elimina transazione
- `GET /api/transactions/stats` - Statistiche

### Sincronizzazione Investimenti

Quando crei una transazione di tipo "Investimento" con ticker, quantità e prezzo:

1. Viene creata una transazione in Trackr
2. Viene automaticamente creato un `Order` nel portfolio pfTrackr
3. L'investimento appare in entrambe le app

## PWA Features

### Installazione

Su mobile: tap "Aggiungi a schermata Home"
Su desktop: click sull'icona di installazione nella barra degli indirizzi

### Offline Mode

- Le chiamate API sono cachate con strategia NetworkFirst
- Cache di 24 ore per le risposte API
- Asset statici cachati automaticamente

### Manifest

- Nome: "Trackr - Gestione Spese"
- Tema: Blue (#0ea5e9)
- Orientamento: Portrait
- Display: Standalone

## Scripts Disponibili

```bash
npm run dev        # Dev server con HMR
npm run build      # Build produzione
npm run preview    # Preview build locale
npm run lint       # Linting ESLint
```

## Compatibilità

- Chrome/Edge 90+
- Safari 14+
- Firefox 88+
- Mobile: iOS 14+, Android 8+

## Note

- Il backend deve essere avviato su `http://localhost:5000` (o modificare `VITE_API_URL`)
- Le icone PWA vanno aggiunte in `public/` (pwa-192x192.png, pwa-512x512.png)
- Per HTTPS in produzione, aggiornare `VITE_API_URL` con l'URL corretto

## Licenza

MIT
