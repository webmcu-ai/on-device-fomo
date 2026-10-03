// sw.js
const myCacheName = "fomo-cache-v1";
const myAssetsToCache = [
  "./",
  "./index.html",
  "./pwa/manifest.json",
  "./pwa/pwa.js",
  "./pwa/icon-192.png",
  "./pwa/icon-512.png"
];

// Install event using wait hooks
self.addEventListener("install", (myEvent) => {
  myEvent.waitUntil((async () => {
    const myCache = await caches.open(myCacheName);
    await myCache.addAll(myAssetsToCache);
    await self.skipWaiting();
  })());
});

// Activate event to clean old caches
self.addEventListener("activate", (myEvent) => {
  myEvent.waitUntil((async () => {
    await self.clients.claim();
  })());
});

// Fetch event using async/await response strategies
self.addEventListener("fetch", (myEvent) => {
  myEvent.respondWith((async () => {
    const myCachedResponse = await caches.match(myEvent.request);
    if (myCachedResponse) {
      return myCachedResponse;
    }
    return fetch(myEvent.request);
  })());
});
