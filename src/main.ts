import * as dataset from "./dataset.js";
import * as model from "./model.js";
import { setBackend } from "./tf.js";
import * as viz from "./viz.js";
import * as uiControls from "./ui-controls.js";
import * as perfectWeights from "./perfect-weights.js";
import "./orchestrators/reinitializeModel.js";
import "./orchestrators/refreshViz.js";
import "./orchestrators/onTrainingStepCompleted.js";
import "./orchestrators/training.js";
import "./orchestrators/setTrainingData.js";

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;

// Define the global orchestrator functions
// These call all module implementations in the correct order

window.reinitializeModel = (newNumLayers: number, newNeuronsPerLayer: number): void => {
  // 1. First, create a new model (model.ts)
  model.reinitializeModel(newNumLayers, newNeuronsPerLayer);

  // 2. Generate new data and push to modules via orchestrator
  dataset.refreshTrainingData();

  // 3. Update visualization (viz.ts)
  viz.reinitializeModel(newNumLayers, newNeuronsPerLayer);

  // 4. Update perfect weights button state (perfect-weights.ts)
  perfectWeights.reinitializeModel(newNumLayers, newNeuronsPerLayer);

  // 5. Update UI controls state (ui-controls.ts)
  uiControls.reinitializeModel(newNumLayers, newNeuronsPerLayer);

  // 6. Set ready status
  viz.setStatusMessage('Ready to train!');
};

window.refreshViz = (): void => {
  viz.refreshViz();
};

window.onTrainingStepCompleted = (epoch: number, loss: number): void => {
  viz.onTrainingStepCompleted(epoch, loss);
};

window.startTraining = (): void => {
  // 1. Start training in model.ts (start training loop)
  model.startTraining();
  
  // 2. Update UI in ui-controls.ts (button text)
  uiControls.startTraining();
};

window.stopTraining = (): void => {
  // 1. Stop training in model.ts (stop training loop)
  model.stopTraining();
  
  // 2. Update UI in ui-controls.ts (button text)
  uiControls.stopTraining();
};

window.setTrainingData = (data): void => {
  // 1. Set training data in model.ts
  model.setTrainingData(data);
  
  // 2. Set training data reference in viz.ts for lookup
  viz.setTrainingData(data);
};

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  // Always stop training (safe to call even if not training)
  window.stopTraining();
  model.disposeTrainingData();
  viz.disposeVizData();
}

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize module DOM references (each module calls document.getElementById directly)
  viz.initVizDom();
  uiControls.initUiControlsDom();
  perfectWeights.initPerfectWeightsDom();

  // Get DOM elements needed only by main.ts for event listeners
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  const numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  const numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  const neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  const neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;

  // Add event listener for backend changes
  backendSelector.addEventListener('change', async () => {
    prepareForReinitialize();
    await setBackend(backendSelector.value);
    window.reinitializeModel(numLayers, neuronsPerLayer);
  });

  // Add event listeners for layer configuration sliders
  numLayersSlider.addEventListener('input', () => {
    numLayers = parseInt(numLayersSlider.value, 10);
    numLayersValue.textContent = numLayers.toString();
    prepareForReinitialize();
    window.reinitializeModel(numLayers, neuronsPerLayer);
  });

  neuronsPerLayerSlider.addEventListener('input', () => {
    neuronsPerLayer = parseInt(neuronsPerLayerSlider.value, 10);
    neuronsPerLayerValue.textContent = neuronsPerLayer.toString();
    prepareForReinitialize();
    window.reinitializeModel(numLayers, neuronsPerLayer);
  });

  // Set up input textbox event listeners in viz module
  viz.setupInputEventListeners();

  // Initial setup
  await setBackend(backendSelector.value);
  window.reinitializeModel(numLayers, neuronsPerLayer);
});
