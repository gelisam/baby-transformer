import { Schedule } from "../messageLoop.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { StartTrainingHandler, StopTrainingHandler, StartTrainingMsg, StopTrainingMsg } from "../messages/training.js";

let trainButton: HTMLButtonElement | null = null;
let isTraining = false;

function getTrainButton(): HTMLButtonElement {
  if (!trainButton) {
    trainButton = document.getElementById('train-button') as HTMLButtonElement;
  }
  return trainButton;
}

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

const reinitializeModel: ReinitializeModelHandler = (_schedule, _numLayers, _neuronsPerLayer, _inputFormat) => {
  updateTrainButtonText();
};

const startTraining: StartTrainingHandler = (_schedule) => {
  isTraining = true;
  updateTrainButtonText();
};

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
