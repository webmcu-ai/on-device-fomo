// ─── webmcu-ai pwa.js ─────────────────────────────────────────────────────────
// Include in your repo's root index.html:
//   <script src="pwa/pwa.js"></script>
//
// Add one button anywhere in index.html:
//   <button id="myAppBtn" onclick="myHandleAppAction()">Install Offline App (Free)</button>
// ─────────────────────────────────────────────────────────────────────────────

// Derive repo name from URL path
const myPwaRepoName = (function() {
    const myParts = window.location.pathname.split('/').filter(Boolean);
    return myParts[0] || "unknown";
})();

let myPwaDeferredPrompt = null;

// ─── Button state ──────────────────────────────────────────────────────────────

function myPwaUpdateBtn() {
    const myBtn = document.getElementById('myAppBtn');
    if (!myBtn) return;

    // Hide button entirely when already running as an installed standalone PWA
    if (window.matchMedia('(display-mode: standalone)').matches) {
        myBtn.style.display = 'none';
        return;
    }
    myBtn.style.display = '';

    if (myPwaDeferredPrompt) {
        myBtn.textContent = "Install Offline App ↓";
    } else {
        myBtn.textContent = "Support / Setup App →";
    }
}

// ─── Button click handler — call this from onclick="myHandleAppAction()" ──────

async function myHandleAppAction() {
    // If the browser is ready to install, trigger the native prompt first
    if (myPwaDeferredPrompt) {
        myPwaDeferredPrompt.prompt();
        const { outcome } = await myPwaDeferredPrompt.userChoice;
        myPwaDeferredPrompt = null;
        myPwaUpdateBtn();
        
        // After they interact with the install prompt, redirect them to the 
        // contribution/support page so they can see how to help out.
        setTimeout(() => {
            window.location.href = './pwa/pwa.html?installed=1';
        }, 1000);
        return;
    }

    // Fallback/Direct click: Send them to the info and support hub
    window.location.href = './pwa/pwa.html';
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
