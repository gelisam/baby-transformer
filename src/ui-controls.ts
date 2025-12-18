/**
 * UI control handlers for model training and configuration.
 * 
 * This module manages user interactions with the training controls
 * and layer configuration sliders.
 */

import { AppState, DomElements } from "./types.js";
import { trainingStep } from "./model.js";
import { updatePerfectWeightsButton } from "./perfect-weights.js";
import { ResourceManager } from "./resource-manager.js";

const resourceManager = new ResourceManager();

/**
 * Toggle between training and paused states.
 * 
 * When starting training, this initiates the training loop.
 * When pausing, it stops the loop but preserves the current state.
 * 
 * @param appState - The application state
 * @param dom - DOM elements for UI updates
 */
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

/**
 * Update the layer configuration and reinitialize the model.
 * 
 * This function is called when the user changes the number of layers
 * or neurons per layer. It stops training, disposes of old tensors,
 * and creates a new model with the updated architecture.
 * 
 * @param appState - The application state
 * @param dom - DOM elements for UI updates
 * @param initializeNewModel - Callback function to create a new model
 * @param numLayers - Number of hidden layers (unused, stored in appState)
 * @param neuronsPerLayer - Neurons per layer (unused, stored in appState)
 */
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
