// ─── webmcu-ai pwa.js ─────────────────────────────────────────────────────────
// Include in your repo's root index.html:
//   <script src="pwa/pwa.js"></script>
//
// Add one button anywhere in your root index.html:
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

    // If running as an installed standalone PWA, change the button to a support invite
    if (window.matchMedia('(display-mode: standalone)').matches) {
        myBtn.textContent = "♥ Support TinyML Education";
        myBtn.style.background = "#e7f3ff"; 
        myBtn.style.color = "#0056b3";
        myBtn.style.border = "1px solid #007bff";
        return;
    }

    // If the browser is primed and ready to install
    if (myPwaDeferredPrompt) {
        myBtn.textContent = "Install Offline App ↓";
        myBtn.style.display = '';
    } else {
        // Fallback or post-install state on standard browser tab
        myBtn.textContent = "Support Project / PWA Info";
    }
}

// ─── Button click handler — called from onclick="myHandleAppAction()" ──────

async function myHandleAppAction() {
    // If running standalone, clicking always goes straight to the Open Collective
    if (window.matchMedia('(display-mode: standalone)').matches) {
        window.open("https://opencollective.com/mlsysbook", "_blank");
        return;
    }

    // If the browser installation prompt is available, trigger it
    if (myPwaDeferredPrompt) {
        myPwaDeferredPrompt.prompt();
        const { outcome } = await myPwaDeferredPrompt.userChoice;
        myPwaDeferredPrompt = null;
        myPwaUpdateBtn();
        return;
    }

    // If no prompt is available (or they already dealt with it), open the support link
    // and print a quick message in the console or alert for clarity.
    const myConfirm = confirm("This app is optimized to work 100% offline. Would you like to visit our Open Collective page to support hardware distribution for the Global South?");
    if (myConfirm) {
        window.open("https://opencollective.com/mlsysbook", "_blank");
    }
}

// ─── Service worker registration ──────────────────────────────────────────────
// Looks for sw.js exactly where you kept it: at the root of the repository

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
