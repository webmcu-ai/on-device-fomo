// ─── webmcu-ai pwa.js ─────────────────────────────────────────────────────────
// Include in your repo's root index.html:
//   <script src="pwa/pwa.js"></script>
//
// Add one button anywhere in index.html:
//   <button id="myAppBtn" onclick="myHandleAppAction()">Install App ($6)</button>
//
// Each repo also needs:
//   pwa/index.html   (payment hub — this folder)
//   pwa/sw.js        (service worker — must be at repo root for full scope)
//   manifest.json    (at repo root)
//   icon-192.png     (at repo root)
//   icon-512.png     (at repo root)
// ─────────────────────────────────────────────────────────────────────────────

const myPwaSalt = "webmcu_pwa_2026";

// Derive repo name from URL path (same logic as pwa/index.html)
const myPwaRepoName = (function() {
    const myParts = window.location.pathname.split('/').filter(Boolean);
    return myParts[0] || "unknown";
})();

let myPwaDeferredPrompt = null;

// ─── Token helpers ─────────────────────────────────────────────────────────────

function myPwaTokenKey() {
    return `_pwa_paid_${myPwaRepoName}`;
}

function myPwaIsPaid() {
    try {
        const myVal = localStorage.getItem(myPwaTokenKey());
        if (!myVal) return false;
        return atob(myVal).includes(`_${myPwaSalt}`);
    } catch(e) { return false; }
}

// No URL token handling needed — pwa/index.html writes directly to localStorage
// (same origin: webmcu-ai.github.io), so the token is already present when
// index.html loads after the redirect.

// ─── Button state ──────────────────────────────────────────────────────────────

function myPwaUpdateBtn() {
    const myBtn = document.getElementById('myAppBtn');
    if (!myBtn) return;

    // Hide button entirely when already running as installed PWA
    if (window.matchMedia('(display-mode: standalone)').matches) {
        myBtn.style.display = 'none';
        return;
    }
    myBtn.style.display = '';

    if (!myPwaIsPaid()) {
        myBtn.textContent = "Install App ($6) →";
    } else if (myPwaDeferredPrompt) {
        myBtn.textContent = "Install App ↓";
    } else {
        myBtn.textContent = "App Purchased ✓";
    }
}

// ─── Button click handler — call this from onclick="myHandleAppAction()" ──────

async function myHandleAppAction() {
    if (!myPwaIsPaid()) {
        // Send user to pwa/index.html, which knows how to return here
        window.location.href = 'pwa/';
        return;
    }

    if (myPwaDeferredPrompt) {
        myPwaDeferredPrompt.prompt();
        const { outcome } = await myPwaDeferredPrompt.userChoice;
        myPwaDeferredPrompt = null;
        myPwaUpdateBtn();
    }
    // If no prompt available: already installed, or browser doesn't support it.
    // Button already shows "App Purchased ✓" — nothing more to do.
}

// ─── Service worker registration ──────────────────────────────────────────────
// sw.js must live at the repo root (e.g. /torchjs/sw.js) so its scope covers
// the whole repo. The pwa/ subfolder can't register a wider-scope SW.

function myPwaRegisterSW() {
    if (!('serviceWorker' in navigator)) return;
    // Use absolute path so this works regardless of which page loads pwa.js
    const mySwPath = `/${myPwaRepoName}/sw.js`;
    navigator.serviceWorker.register(mySwPath, { scope: `/${myPwaRepoName}/` })
        .catch(err => console.warn('pwa.js: SW registration failed', err));
}

// ─── Init on load ──────────────────────────────────────────────────────────────

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    myPwaDeferredPrompt = e;
    myPwaUpdateBtn();
});

window.addEventListener('appinstalled', () => {
    myPwaDeferredPrompt = null;
    myPwaUpdateBtn();
});

window.addEventListener('load', () => {
    myPwaRegisterSW();
    myPwaUpdateBtn();
});
