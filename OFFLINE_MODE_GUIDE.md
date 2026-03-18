# Guida: Trackr PWA in Modalità Offline

## Cosa abbiamo fatto

Abbiamo trasformato Trackr PWA da un'app che **richiede sempre un server** a un'app **completamente offline-first** che salva tutti i dati localmente sul dispositivo.

## Cambiamenti Principali

### 1. Storage Locale con IndexedDB

**File creati:**
- `trackr-pwa/src/services/db.ts` - Gestione database IndexedDB
- `trackr-pwa/src/services/localStorage.ts` - Service layer per operazioni offline

**Stores creati:**
- `users` - Utenti locali
- `transactions` - Transazioni
- `categories` - Categorie
- `subcategories` - Sottocategorie
- `accounts` - Account bancari
- `portfolios` - Portfolio investimenti

### 2. API Service Ibrido

**File modificato:** `trackr-pwa/src/services/api.ts`

Ora supporta due modalità:
```typescript
const OFFLINE_MODE = import.meta.env.VITE_OFFLINE_MODE !== 'false';
```

- `OFFLINE_MODE=true` → Usa IndexedDB locale
- `OFFLINE_MODE=false` → Usa chiamate HTTP al backend

### 3. Pagina Impostazioni

**File creato:** `trackr-pwa/src/pages/SettingsPage.tsx`

Funzionalità:
- Visualizza info utente
- **Export Backup** - Scarica JSON con tutti i dati
- **Import Backup** - Ripristina da file JSON
- Istruzioni installazione PWA
- Logout

**Route aggiunta:** `/settings` (accessibile dall'icona ingranaggio nell'header)

### 4. Configurazione

**File creati:**
- `.env` - Modalità offline abilitata di default
- `.env.example` - Template per configurazione

```env
VITE_OFFLINE_MODE=true
VITE_API_URL=http://localhost:8000
```

### 5. Documentazione

**File aggiornato:** `trackr-pwa/README.md`

Documenta:
- Modalità offline e online
- Come fare backup
- Come installare l'app (Android/iOS/Desktop)
- Come convertire in APK con Capacitor

## Come Usare

### Modalità Offline (Default)

1. **Avvia l'app:**
   ```bash
   cd trackr-pwa
   npm run dev
   ```

2. **Login:**
   - Inserisci qualsiasi username/password
   - Viene creato automaticamente un utente locale

3. **Usa normalmente:**
   - Tutti i dati sono salvati in IndexedDB
   - Funziona anche senza connessione internet

4. **Backup periodici:**
   - Vai su Impostazioni → Esporta Backup
   - Salva il file JSON in un posto sicuro

### Modalità Online (Se serve)

1. **Configura `.env`:**
   ```env
   VITE_OFFLINE_MODE=false
   VITE_API_URL=http://localhost:8000
   ```

2. **Avvia il backend:**
   ```bash
   cd backend
   python main.py
   ```

3. **Usa l'app normalmente** - I dati vanno sul server

## Installare come App Mobile

### Android

1. Apri Chrome e vai su `http://localhost:5175` (o IP della rete locale)
2. Menu (3 puntini) → "Aggiungi a schermata Home"
3. L'app funziona come app nativa, anche offline

### iOS

1. Apri Safari e vai all'app
2. Pulsante "Condividi" → "Aggiungi a Home"
3. L'app appare sulla home screen

### Desktop

1. Chrome/Edge: clicca icona installazione nella barra
2. L'app si apre in una finestra dedicata

## Creare APK Nativo (Opzionale)

Se vuoi distribuire un file `.apk` installabile:

```bash
# Installa Capacitor
npm install @capacitor/core @capacitor/cli @capacitor/android

# Setup
npx cap init
npx cap add android

# Build
npm run build
npx cap sync

# Apri Android Studio
npx cap open android
```

Da Android Studio:
- Build → Generate Signed Bundle/APK
- Scegli APK
- Firma con tuo keystore
- L'APK sarà in `android/app/build/outputs/apk/`

## Vantaggi Modalità Offline

✅ **Zero costi server** - Non serve mantenere un backend acceso
✅ **Privacy totale** - Dati solo sul tuo dispositivo
✅ **Funziona sempre** - Anche senza internet
✅ **Velocità massima** - Nessuna latenza di rete
✅ **Semplice** - Niente configurazione database/server

## Sincronizzazione Futura (Opzionale)

Per gli investimenti o per sincronizzare tra dispositivi:

### Opzione 1: Raspberry Pi come DB
- Installa PostgreSQL sul Raspberry Pi
- Configura port forwarding o VPN
- Usa modalità online quando a casa

### Opzione 2: Push Manuale
- Tieni app offline per spese quotidiane
- Quando vuoi sincronizzare investimenti:
  - Export backup
  - Chiamata API a pfTrackr per inviare dati
  - (Da implementare)

### Opzione 3: Sync Automatica
- Background sync quando disponibile connessione
- Conflict resolution
- (Da implementare - più complesso)

## Backup e Sicurezza

⚠️ **IMPORTANTE**: I dati in IndexedDB sono persistenti MA:

- Se cancelli i dati del browser, **perdi tutto**
- Se usi un altro browser, **non vedi i dati**
- Se cambi dispositivo, **devi importare il backup**

**Raccomandazione forte**: Fai export backup settimanale e salvalo su:
- Cloud (Google Drive, Dropbox, ecc.)
- USB/SD card
- Computer

## Domande Frequenti

**Q: Posso usare l'app su più dispositivi?**
A: Sì, ma devi esportare e importare i backup manualmente. Non c'è sync automatica in modalità offline.

**Q: Quanto spazio ho a disposizione?**
A: IndexedDB ha circa 50-100MB di spazio (dipende dal browser). Più che sufficiente per anni di spese.

**Q: Cosa succede se aggiorno il browser?**
A: I dati rimangono. Vengono persi solo se cancelli esplicitamente i dati del sito.

**Q: Posso passare da offline a online?**
A: Sì, cambia `VITE_OFFLINE_MODE` nel `.env`. Ma i dati offline e online sono separati.

**Q: L'app funziona anche senza WiFi?**
A: Sì! Completamente. Una volta caricata la prima volta, funziona sempre offline.

## Prossimi Passi Consigliati

1. **Testa l'app** - Crea transazioni, categorie, account
2. **Installa come PWA** - Prova su Android/iOS
3. **Fai un backup** - Test della funzione export/import
4. **Usa quotidianamente** - Traccia spese per 1-2 settimane
5. **Considera APK nativo** - Se preferisci app "vera" invece di PWA

Per gli investimenti:
- Decidi se vuoi Raspberry Pi sempre online
- Oppure sync manuale quando serve
- Oppure tieni separato pfTrackr per investimenti

## Supporto

File principali da conoscere:
- `trackr-pwa/src/services/db.ts` - Logica database
- `trackr-pwa/src/services/localStorage.ts` - Operazioni CRUD
- `trackr-pwa/src/services/api.ts` - Switch offline/online
- `trackr-pwa/src/pages/SettingsPage.tsx` - Backup/restore

**Buon tracking delle spese!** 💰
