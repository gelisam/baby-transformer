import { ReinitializeModelImpl } from "./orchestrators/reinitializeModel.js";
import { ToggleTrainingImpl } from "./orchestrators/toggleTraining.js";
import { toggleTrainingImpl, getIsTraining } from "./model.js";

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

// Toggle training and update button text
function toggleTrainingMode() {
  toggleTrainingImpl();
  updateTrainButtonText();
}

function updateTrainButtonText() {
  if (trainButton) {
    trainButton.innerText = getIsTraining() ? 'Pause' : 'Train Model';
  }
}

// Implementation for the reinitializeModel orchestrator
const reinitializeModel: ReinitializeModelImpl = (numLayers, neuronsPerLayer) => {
  // Reset button text when model is reinitialized
  updateTrainButtonText();
};

// Implementation for toggleTraining orchestrator
const toggleTraining: ToggleTrainingImpl = () => {
  toggleTrainingMode();
};

export { 
  initUiControlsDom,
  toggleTrainingMode, 
  reinitializeModel,
  toggleTraining
};
