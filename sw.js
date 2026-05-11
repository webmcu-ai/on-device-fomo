// ─── webmcu-ai repo service worker ───────────────────────────────────────────
// Copy this file verbatim to the root of each repo.
// It is intentionally generic — the repo name is inferred from scope.
// Update myCacheName version string when you want to force a cache refresh.
// ─────────────────────────────────────────────────────────────────────────────

const myCacheName = 'webmcu-repo-v1';

// Core assets to cache on install.
// '.' caches the root index.html via the SW scope.
const myStaticAssets = [
    '.',
    'index.html',
];

// ─── Install: cache core assets ───────────────────────────────────────────────

self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(myCacheName)
            .then(cache => cache.addAll(myStaticAssets))
            .then(() => self.skipWaiting())
    );
});

// ─── Activate: remove old caches ─────────────────────────────────────────────

self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(myKeys =>
            Promise.all(
                myKeys
                    .filter(key => key !== myCacheName)
                    .map(key => caches.delete(key))
            )
        ).then(() => self.clients.claim())
    );
});

// ─── Fetch: cache-first with network fallback ─────────────────────────────────

self.addEventListener('fetch', (e) => {
    // Only handle GET requests; skip cross-origin (e.g. GitHub API calls)
    if (e.request.method !== 'GET') return;
    const myUrl = new URL(e.request.url);
    if (myUrl.origin !== self.location.origin) return;

    e.respondWith(
        caches.match(e.request).then(cached => {
            if (cached) return cached;
            return fetch(e.request).then(response => {
                // Cache successful same-origin responses dynamically
                if (response.ok) {
                    const myClone = response.clone();
                    caches.open(myCacheName).then(cache => cache.put(e.request, myClone));
                }
                return response;
            });
        })
    );
});
