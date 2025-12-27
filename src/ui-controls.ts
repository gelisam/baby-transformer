import { ReinitializeModel } from "./orchestrators/reinitializeModel.js";
import { ToggleTraining } from "./orchestrators/toggleTraining.js";
import { getIsTraining } from "./model.js";

// Module-local state for DOM elements (initialized on first use)
let trainButton: HTMLButtonElement | null = null;
let domInitialized = false;

// Initialize DOM elements by calling document.getElementById directly
function initUiControlsDom() {
  if (domInitialized) return;
  trainButton = document.getElementById('train-button') as HTMLButtonElement;
  
  // Set up event listener
  if (trainButton) {
    trainButton.addEventListener('click', () => window.toggleTraining());
  }
  
  domInitialized = true;
}

function updateTrainButtonText() {
  if (trainButton) {
    trainButton.innerText = getIsTraining() ? 'Pause' : 'Train Model';
  }
}

// Implementation for the reinitializeModel orchestrator
const reinitializeModel: ReinitializeModel = (numLayers, neuronsPerLayer) => {
  // Reset button text when model is reinitialized
  updateTrainButtonText();
};

// Implementation for toggleTraining orchestrator
// This module only handles the UI update (button text)
const toggleTraining: ToggleTraining = () => {
  updateTrainButtonText();
};

export { 
  initUiControlsDom,
  reinitializeModel,
  toggleTraining
};
