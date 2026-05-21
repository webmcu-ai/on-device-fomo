// pwa/pwa.js

let mySavedInstallPromptEvent = null;

// Register the service worker using async/await
async function myRegisterServiceWorker() {
if ("serviceWorker" in navigator) {
    try {
      // Automatically extracts the repository name (e.g., "/on-device-fomo/")
      const myRepoScope = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);
      
      const myRegistration = await navigator.serviceWorker.register(`${myRepoScope}sw.js`, {
        scope: myRepoScope
      });
      console.log("Dynamic PWA registered on scope:", myRegistration.scope);
    } catch (myError) {
      console.error("PWA registration failed:", myError);
    }
  }
}

// Listen for the native browser install availability prompt
window.addEventListener("beforeinstallprompt", (myEvent) => {
  // Prevent older platforms from automatically showing the prompt
  myEvent.preventDefault();
  
  // Stash the event so it can be triggered later
  mySavedInstallPromptEvent = myEvent;
  
  // Make our custom install button visible
  const myInstallButton = document.getElementById("myPwaInstallButton");
  if (myInstallButton) {
    myInstallButton.style.display = "inline-block";
  }
});

// Triggered via inline onclick handler from index.html
async function myHandleInstallButtonClick() {
  if (!mySavedInstallPromptEvent) {
    return;
  }
  
  // Show the native install prompt
  mySavedInstallPromptEvent.prompt();
  
  // Wait for the user's choice using async/await
  const myUserChoice = await mySavedInstallPromptEvent.userChoice;
  console.log("User installation choice outcome:", myUserChoice.outcome);
  
  // Clear the prompt event resource since it can only be used once
  mySavedInstallPromptEvent = null;
  
  // Hide the button again
  const myInstallButton = document.getElementById("myPwaInstallButton");
  if (myInstallButton) {
    myInstallButton.style.display = "none";
  }
}

// Reset if already installed
window.addEventListener("appinstalled", () => {
  console.log("Application successfully installed!");
  mySavedInstallPromptEvent = null;
  const myInstallButton = document.getElementById("myPwaInstallButton");
  if (myInstallButton) {
    myInstallButton.style.display = "none";
  }
});

// Execute registration
myRegisterServiceWorker();
