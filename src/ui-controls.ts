import { AppState } from "./types.js";
import { trainingStep } from "./model.js";

async function toggleTrainingMode(appState: AppState) {
  appState.isTraining = !appState.isTraining;
  const trainButton = document.getElementById('train-button')!;

  if (appState.isTraining) {
    trainButton.innerText = 'Pause';
    requestAnimationFrame(() => trainingStep(appState));
  } else {
    trainButton.innerText = 'Train Model';
  }
}

export { toggleTrainingMode };
