# Trackr PWA - Gestione Spese Offline-First

Progressive Web App per tracciare le tue spese personali, **completamente funzionante offline** senza bisogno di server.

## Caratteristiche

- **100% Offline**: Tutti i dati salvati localmente su IndexedDB
- **PWA Installabile**: Funziona come app nativa su Android, iOS e desktop
- **Zero dipendenze server**: Non serve backend per il funzionamento normale
- **Backup/Restore**: Esporta e importa i tuoi dati in formato JSON
- **Mobile-First**: UI ottimizzata per dispositivi touch
- **Dark Mode**: Supporto automatico per modalità scura
- **Modalità Online (opzionale)**: Supporto per backend se necessario

## Tecnologie

- **React 18** + **TypeScript**
- **Vite** (build tool)
- **Tailwind CSS** (styling)
- **IndexedDB** (database locale)
- **Vite PWA Plugin** (service worker + manifest)
- **React Router** (navigation)
- **Axios** (API client per modalità online)

## Setup

### 1. Installazione dipendenze

```bash
cd trackr-pwa
npm install
```

### 2. Configurazione ambiente

Il file `.env` è già configurato per modalità offline (default):

```env
VITE_OFFLINE_MODE=true
VITE_API_URL=http://localhost:8000
```

**Per usare un backend server invece** (opzionale):
```env
VITE_OFFLINE_MODE=false
VITE_API_URL=http://localhost:8000  # URL del tuo backend
```

### 3. Avvio server di sviluppo

```bash
npm run dev
```

L'app sarà disponibile su [http://localhost:5174](http://localhost:5174) (o porta successiva se occupata)

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

## Modalità di Funzionamento

### Modalità Offline (Default)

L'app usa **IndexedDB** per salvare tutti i dati localmente:

- Transazioni (spese, entrate, investimenti, trasferimenti)
- Categorie e sottocategorie
- Account
- Portfolio

**Vantaggi**:
- Nessun costo server
- Funziona sempre, anche senza internet
- Privacy totale (dati solo sul tuo dispositivo)
- Velocità massima

**Backup**: Vai su Impostazioni → Esporta Backup (file JSON)

### Modalità Online (Opzionale)

Imposta `VITE_OFFLINE_MODE=false` nel file `.env` per usare un backend.

## Funzionalità

### Autenticazione

- Login automatico in modalità offline (utente locale)
- Login con username/password se backend attivo
- Logout

### Account

- Gestione conti bancari/portafogli
- Bilancio calcolato automaticamente
- Icone personalizzate

### Categorie

- Gestione categorie con sottocategorie
- Statistiche per categoria
- Icone personalizzate

### Transazioni

- Lista completa di tutte le transazioni
- Filtri per data, categoria, tipo
- Modifica ed eliminazione
- Supporto investimenti (ticker, quantità, prezzo)
- Supporto trasferimenti tra conti

### Statistiche (Recap)

- Vista per periodo personalizzato
- Distribuzione per categoria
- Trend mensile
- Grafici delle spese

### Portfolio

- Gestione portfolio investimenti
- Integrazione futura con pfTrackr

### Impostazioni

- Visualizzazione info utente
- **Export Backup**: Scarica tutti i dati in JSON
- **Import Backup**: Ripristina dati da backup precedente
- Istruzioni installazione PWA
- Logout

## Struttura Dati IndexedDB

### Stores

- `users`: Informazioni utente locale
- `transactions`: Tutte le transazioni
- `categories`: Categorie con sottocategorie
- `subcategories`: Sottocategorie
- `accounts`: Account finanziari
- `portfolios`: Portfolio di investimenti

### Export Format

Il backup è un file JSON:

```json
{
  "version": 1,
  "exportDate": "2026-03-11T...",
  "userId": "local-user",
  "data": {
    "transactions": [...],
    "categories": [...],
    "accounts": [...],
    "portfolios": [...]
  }
}
```

## Installare come App

### Android
1. Apri l'app nel browser Chrome
2. Tocca i 3 puntini → "Aggiungi a schermata Home"
3. L'app funzionerà come un'app nativa

### iOS
1. Apri l'app in Safari
2. Tocca "Condividi" → "Aggiungi a Home"
3. L'app apparirà sulla home screen

### Desktop (Chrome/Edge)
1. Clicca l'icona di installazione nella barra degli indirizzi
2. Conferma l'installazione

## Convertire in APK (Opzionale)

Per creare un file APK nativo:

```bash
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init
npx cap add android
npm run build
npx cap sync
npx cap open android
```

Poi genera l'APK da Android Studio.

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

## Sicurezza Dati

I dati sono salvati nel browser (IndexedDB):

- **Pro**: Privacy totale, nessun dato in rete
- **Con**: Se cancelli i dati del browser, perdi tutto

**Raccomandazione**: Fai backup regolari con la funzione Export!

## Limiti IndexedDB

- ~50-100MB di spazio (dipende dal browser)
- Persistenti finché non cancellati manualmente
- Non condivisi tra browser
- Non sincronizzati tra dispositivi

## File Principali

- [src/services/db.ts](src/services/db.ts): Gestione IndexedDB
- [src/services/localStorage.ts](src/services/localStorage.ts): Service layer dati locali
- [src/services/api.ts](src/services/api.ts): API service con switch offline/online
- [src/pages/SettingsPage.tsx](src/pages/SettingsPage.tsx): Export/Import backup

## Licenza

MIT
