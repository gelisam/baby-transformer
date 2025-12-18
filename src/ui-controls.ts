import { AppState, DomElements } from "./types.js";
import { trainingStep } from "./model.js";
import { updatePerfectWeightsButton } from "./perfect-weights.js";

async function toggleTrainingMode(appState: AppState, dom: DomElements) {
  appState.isTraining = !appState.isTraining;
  const trainButton = dom.trainButton;

  if (appState.isTraining) {
    trainButton.innerText = 'Pause';
    requestAnimationFrame(() => trainingStep(appState, dom));
  } else {
    trainButton.innerText = 'Train Model';
  }
}

function updateLayerConfiguration(appState: AppState, dom: DomElements, initializeApp: () => void): void {
  // Stop training and reinitialize model
  if (appState.isTraining) {
    toggleTrainingMode(appState, dom); // Toggles isTraining to false
  }

  initializeApp();
  updatePerfectWeightsButton(appState, dom);
}

export { toggleTrainingMode, updateLayerConfiguration };
