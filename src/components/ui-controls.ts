import { Schedule } from "../messageLoop.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { StartTrainingHandler, StopTrainingHandler, StartTrainingMsg, StopTrainingMsg } from "../messages/training.js";

// Module-local state for DOM elements (initialized on first use)
let trainButton: HTMLButtonElement | null = null;

// Module-local state for training status
let isTraining = false;

// Getter function that checks and initializes DOM element if needed
function getTrainButton(): HTMLButtonElement {
  if (!trainButton) {
    trainButton = document.getElementById('train-button') as HTMLButtonElement;
  }
  return trainButton;
}

// Handler for the Init message - attach event listeners
const init: InitHandler = (_schedule) => {
  const button = getTrainButton();
  button.addEventListener('click', () => {
    if (isTraining) {
      window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);
    } else {
      window.messageLoop({ type: "StartTraining" } as StartTrainingMsg);
    }
  });
};

function updateTrainButtonText() {
  const button = getTrainButton();
  button.innerText = isTraining ? 'Pause' : 'Train Model';
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, _numLayers, _neuronsPerLayer, _inputFormat, _vocabSize) => {
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
  init,
  reinitializeModel,
  startTraining,
  stopTraining
};
