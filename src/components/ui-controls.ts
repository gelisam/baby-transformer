import { Schedule } from "../messageLoop.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { StartTrainingHandler, StopTrainingHandler, StartTrainingMsg, StopTrainingMsg } from "../messages/training.js";

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
        window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);
      } else {
        window.messageLoop({ type: "StartTraining" } as StartTrainingMsg);
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

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, _numLayers, _neuronsPerLayer, _inputFormat) => {
  // Reset button text when model is reinitialized
  updateTrainButtonText();
};

// Implementation for startTraining message handler
// This module updates its local state and button text
const startTraining: StartTrainingHandler = (_schedule) => {
  isTraining = true;
  updateTrainButtonText();
};

// Implementation for stopTraining message handler
// This module updates its local state and button text
const stopTraining: StopTrainingHandler = (_schedule) => {
  isTraining = false;
  updateTrainButtonText();
};

export { 
  initUiControlsDom,
  reinitializeModel,
  startTraining,
  stopTraining
};
