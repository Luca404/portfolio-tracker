# Guida Deploy - Trackr PWA

## 🚀 Opzioni di Deploy

### Opzione 1: Vercel (CONSIGLIATO - Gratis e Veloce)

1. **Installa Vercel CLI** (solo la prima volta):
```bash
npm install -g vercel
```

2. **Build e Deploy**:
```bash
cd trackr-pwa
npm run build
vercel --prod
```

3. **Segui le istruzioni**:
   - Login con GitHub/GitLab/Email
   - Conferma il progetto
   - Ottieni URL tipo: `https://trackr-pwa.vercel.app`

**Vantaggi**:
- ✅ HTTPS automatico
- ✅ CDN globale
- ✅ Deploy automatici da Git
- ✅ Gratis per progetti personali
- ✅ PWA standalone funziona perfettamente

---

### Opzione 2: Netlify

1. **Installa Netlify CLI**:
```bash
npm install -g netlify-cli
```

2. **Deploy**:
```bash
cd trackr-pwa
npm run build
netlify deploy --prod --dir=dist
```

---

### Opzione 3: GitHub Pages

1. **Installa gh-pages**:
```bash
npm install -D gh-pages
```

2. **Aggiungi al package.json**:
```json
{
  "scripts": {
    "deploy": "npm run build && gh-pages -d dist"
  }
}
```

3. **Deploy**:
```bash
npm run deploy
```

4. **Configura repo**:
   - Settings → Pages → Source: gh-pages branch

**Nota**: Devi configurare `base` in `vite.config.ts`:
```ts
export default defineConfig({
  base: '/nome-repo/',
  // ...
})
```

---

### Opzione 4: Server Proprio (VPS)

#### Con Nginx:

1. **Carica i file**:
```bash
cd trackr-pwa
npm run build
scp -r dist/* user@server:/var/www/trackr/
```

2. **Configura Nginx** (`/etc/nginx/sites-available/trackr`):
```nginx
server {
    listen 443 ssl http2;
    server_name trackr.tuodominio.com;

    # Certificato SSL (usa certbot per Let's Encrypt gratuito)
    ssl_certificate /etc/letsencrypt/live/tuodominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tuodominio.com/privkey.pem;

    root /var/www/trackr;
    index index.html;

    # Caching per asset statici
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Service worker (no cache)
    location ~* (sw\.js|workbox-.*\.js)$ {
        add_header Cache-Control "no-cache";
        expires 0;
    }

    # Routing client-side
    location / {
        try_files $uri $uri/ /index.html;
    }
}

# Redirect HTTP → HTTPS
server {
    listen 80;
    server_name trackr.tuodominio.com;
    return 301 https://$server_name$request_uri;
}
```

3. **Attiva e riavvia**:
```bash
sudo ln -s /etc/nginx/sites-available/trackr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

4. **Certificato SSL gratis**:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d trackr.tuodominio.com
```

---

## 📱 Test PWA Standalone

Dopo il deploy su HTTPS:

1. **Apri l'URL dal telefono** (Chrome/Edge)
2. **Menu → "Installa app"** o banner automatico
3. **Aggiungi a Home**
4. **Apri dalla home** → NON deve esserci la barra del browser ✓

---

## 🔄 Workflow Consigliato

### Development (locale):
```bash
npm run dev
# Testa su http://localhost:5174 (con barra browser = normale)
```

### Production (deploy):
```bash
npm run build
vercel --prod
# Testa su https://trackr-xxx.vercel.app (standalone mode ✓)
```

---

## ⚙️ Variabili d'Ambiente

### Development (`.env` o `.env.local`):
```env
VITE_OFFLINE_MODE=true
VITE_API_URL=http://localhost:8000
```

### Production (`.env.production`):
```env
VITE_OFFLINE_MODE=true
VITE_API_URL=https://api.tuodominio.com
```

---

## 🐛 Troubleshooting

### La PWA non si installa:
- ✅ Verifica HTTPS (http:// NON funziona, solo https://)
- ✅ Controlla manifest su Chrome DevTools → Application → Manifest
- ✅ Verifica service worker registrato

### L'app mostra ancora la barra del browser:
- ✅ Disinstalla completamente dalla home
- ✅ Pulisci cache del browser
- ✅ Reinstalla
- ✅ Verifica che `display: "standalone"` sia nel manifest

### Icone non appaiono:
- ✅ Verifica che icon-192.png e icon-512.png esistano in `public/`
- ✅ Rebuild: `npm run build`

---

## 📊 Checklist Pre-Deploy

- [ ] Build senza errori: `npm run build`
- [ ] Icone presenti: `public/icon-192.png` e `public/icon-512.png`
- [ ] Manifest configurato: `display: "standalone"`
- [ ] Service worker funzionante
- [ ] Variabili d'ambiente di produzione configurate
- [ ] Deploy su piattaforma HTTPS

---

## 💡 Consigli

1. **Usa Vercel/Netlify** per la massima semplicità
2. **HTTPS è obbligatorio** per PWA standalone
3. **Testa sempre su device reale** dopo deploy
4. **La modalità offline** funziona perfettamente con IndexedDB
5. **Non serve backend** se usi `VITE_OFFLINE_MODE=true`
