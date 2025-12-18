import { AppState, DomElements } from "./types.js";
import { trainingStep } from "./model.js";
import { updatePerfectWeightsButton } from "./perfect-weights.js";
import { ResourceManager } from "./resource-manager.js";

const resourceManager = new ResourceManager();

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

function updateLayerConfiguration(appState: AppState, dom: DomElements, initializeNewModel: (dom: DomElements) => void, numLayers: number, neuronsPerLayer: number): void {
  // Stop training and reinitialize model
  if (appState.isTraining) {
    toggleTrainingMode(appState, dom); // Toggles isTraining to false
  }
  
  // Safely dispose tensors using ResourceManager
  resourceManager.disposeTrainingData(appState.data);
  resourceManager.disposeTrainingData(appState.vizData);

  initializeNewModel(dom);
  updatePerfectWeightsButton(appState, dom);
}

export { toggleTrainingMode, updateLayerConfiguration };
