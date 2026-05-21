// pwa/pwa.js

// Global variable to store the installation event
let mySavedBeforeInstallPromptEvent = null;

// Register the service worker using async/await
async function myRegisterServiceWorkerFunction() {
  if ("serviceWorker" in navigator) {
    try {
      const myRegistrationDetails = await navigator.serviceWorker.register("sw.js");
      console.log("Service Worker registered successfully:", myRegistrationDetails);
    } catch (myRegistrationError) {
      console.error("Service Worker registration failed:", myRegistrationError);
    }
  }
}

// Listen for the browser's native install prompt challenge
window.addEventListener("beforeinstallprompt", (myEventDetails) => {
  // Prevent the older default browser install banner from showing automatically
  myEventDetails.preventDefault();
  
  // Save the event so we can trigger it later
  mySavedBeforeInstallPromptEvent = myEventDetails;
  
  // Reveal your custom install button
  const myInstallButtonElement = document.getElementById("myPwaInstallButton");
  if (myInstallButtonElement) {
    myInstallButtonElement.style.display = "inline-block";
  }
});

// Function called when your custom button is clicked
async function myHandleInstallButtonClick() {
  if (!mySavedBeforeInstallPromptEvent) {
    return;
  }
  
  // Show the native browser installation prompt
  mySavedBeforeInstallPromptEvent.prompt();
  
  // Wait for the user to respond to the prompt
  const myUserChoiceResult = await mySavedBeforeInstallPromptEvent.userChoice;
  console.log("User selection outcome:", myUserChoiceResult.outcome);
  
  // Clear the saved event variable since it can only be used once
  mySavedBeforeInstallPromptEvent = null;
  
  // Hide the custom install button again
  const myInstallButtonElement = document.getElementById("myPwaInstallButton");
  if (myInstallButtonElement) {
    myInstallButtonElement.style.display = "none";
  }
}

// Automatically listen for successful installation completion
window.addEventListener("appinstalled", () => {
  console.log("PWA application was successfully installed on the device!");
});

// Run the service worker registration
myRegisterServiceWorkerFunction();
