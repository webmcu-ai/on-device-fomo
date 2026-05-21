// ─── webmcu-ai pwa.js ─────────────────────────────────────────────────────────
// Include in your repo's root index.html:
//   <script src="pwa/pwa.js"></script>
//
// Add one button anywhere in your root index.html:
//   <button id="myAppBtn" onclick="myHandleAppAction()">Install Offline App (Free)</button>
// ─────────────────────────────────────────────────────────────────────────────

const myPwaRepoName = (function() {
    const myParts = window.location.pathname.split('/').filter(Boolean);
    return myParts[0] || "unknown";
})();

let myPwaDeferredPrompt = null;

// ─── Button state ──────────────────────────────────────────────────────────────

function myPwaUpdateBtn() {
    const myBtn = document.getElementById('myAppBtn');
    if (!myBtn) return;

    // If running as an installed standalone PWA, update text and style for support
    if (window.matchMedia('(display-mode: standalone)').matches) {
        myBtn.textContent = "♥ Support TinyML Education";
        myBtn.style.background = "#e7f3ff"; 
        myBtn.style.color = "#0056b3";
        myBtn.style.border = "1px solid #007bff";
        return;
    }

    // Dynamic state based on browser installation readiness
    if (myPwaDeferredPrompt) {
        myBtn.textContent = "Install Offline App ↓";
    } else {
        myBtn.textContent = "Support Project / Open Collective";
    }
}

// ─── Button click handler — called from onclick="myHandleAppAction()" ──────

async function myHandleAppAction() {
    // If running standalone, go straight to the Open Collective page
    if (window.matchMedia('(display-mode: standalone)').matches) {
        window.open("https://opencollective.com/mlsysbook", "_blank");
        return;
    }

    // If the browser installation prompt is active, trigger it immediately
    if (myPwaDeferredPrompt) {
        myPwaDeferredPrompt.prompt();
        const { outcome } = await myPwaDeferredPrompt.userChoice;
        myPwaDeferredPrompt = null;
        myPwaUpdateBtn();
        return;
    }

    // Direct fallback: if no prompt is available, open the link instantly
    window.open("https://opencollective.com/mlsysbook", "_blank");
}

// ─── Service worker registration ──────────────────────────────────────────────

function myPwaRegisterSW() {
    if (!('serviceWorker' in navigator)) return;
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
