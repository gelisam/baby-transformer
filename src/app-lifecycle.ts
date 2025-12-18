import { generateData } from "./dataset.js";
import { createModel } from "./model.js";
import { pickRandomInputs, drawViz, drawNetworkArchitecture } from "./viz.js";
import { AppState, DomElements, TrainingData } from "./types.js";

function disposeTrainingData(data?: TrainingData): void {
  if (!data) {
    return;
  }
  try {
    data.inputTensor.dispose();
  } catch (e) {
    // Tensor may already be disposed
  }
  try {
    data.outputTensor.dispose();
  } catch (e) {
    // Tensor may already be disposed
  }
}

function disposeAppState(appState: AppState): void {
  if (appState.model) {
    appState.model.dispose();
  }
  disposeTrainingData(appState.data);
  disposeTrainingData(appState.vizData);
}

function bootstrapAppState(appState: AppState, dom: DomElements): void {
  appState.model = createModel(appState.num_layers, appState.neurons_per_layer);
  appState.data = generateData();
  appState.vizData = pickRandomInputs(appState.data, dom);

  appState.currentEpoch = 0;
  appState.lossHistory.length = 0;
  dom.statusElement.innerHTML = 'Ready to train!';

  drawViz(appState, appState.vizData, dom);
  drawNetworkArchitecture(appState, dom);
}

function resetAppState(appState: AppState, dom: DomElements): void {
  disposeAppState(appState);
  bootstrapAppState(appState, dom);
}

export { disposeAppState, bootstrapAppState, resetAppState };
