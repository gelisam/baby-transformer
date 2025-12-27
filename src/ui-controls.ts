import { ReinitializeModel } from "./orchestrators/reinitializeModel.js";
import { StartTraining } from "./orchestrators/startTraining.js";
import { StopTraining } from "./orchestrators/stopTraining.js";
import "./orchestrators/startTraining.js";
import "./orchestrators/stopTraining.js";

// Module-local state for DOM elements (initialized on first use)
let trainButton: HTMLButtonElement | null = null;
let domInitialized = false;

// Module-local state for training status
let isTraining = false;

// Initialize DOM elements by calling document.getElementById directly
function initUiControlsDom() {
  if (domInitialized) return;
  trainButton = document.getElementById('train-button') as HTMLButtonElement;
  
  // Set up event listener - toggle logic lives here in the UI module
  if (trainButton) {
    trainButton.addEventListener('click', () => {
      if (isTraining) {
        window.stopTraining();
      } else {
        window.startTraining();
      }
    });
  }
  
  domInitialized = true;
}

function updateTrainButtonText() {
  if (trainButton) {
    trainButton.innerText = isTraining ? 'Pause' : 'Train Model';
  }
}

// Implementation for the reinitializeModel orchestrator
const reinitializeModel: ReinitializeModel = (_numLayers, _neuronsPerLayer) => {
  // Reset button text when model is reinitialized
  updateTrainButtonText();
};

// Implementation for startTraining orchestrator
// This module updates its local state and button text
const startTraining: StartTraining = () => {
  isTraining = true;
  updateTrainButtonText();
};

// Implementation for stopTraining orchestrator
// This module updates its local state and button text
const stopTraining: StopTraining = () => {
  isTraining = false;
  updateTrainButtonText();
};

export { 
  initUiControlsDom,
  reinitializeModel,
  startTraining,
  stopTraining
};
