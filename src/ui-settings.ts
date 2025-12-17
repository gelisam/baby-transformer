import { AppState } from "./types.js";
import { toggleTrainingMode } from "./ui-controls.js";
import { updatePerfectWeightsButton } from "./perfect-weights.js";

function updateLayerConfiguration(appState: AppState, initializeNewModel: () => void, numLayers: number, neuronsPerLayer: number): void {
  // Stop training and reinitialize model
  if (appState.isTraining) {
    toggleTrainingMode(appState); // Toggles isTraining to false
  }
  if (appState.data) {
    try {
      appState.data.inputTensor.dispose();
      appState.data.outputTensor.dispose();
    } catch (e) {
      // Tensors may already be disposed
    }
  }
  if (appState.vizData) {
    try {
      appState.vizData.inputTensor.dispose();
      appState.vizData.outputTensor.dispose();
    } catch (e) {
      // Tensors may already be disposed
    }
  }

  initializeNewModel();
  updatePerfectWeightsButton(appState);
}

export { updateLayerConfiguration };
