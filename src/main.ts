import { generateData, getTrainingInputArray, getTrainingOutputArray, getTrainingInputTensor, getTrainingOutputTensor, disposeTrainingData } from "./dataset.js";
import { reinitializeModel as modelReinitialize, toggleTraining as modelToggleTraining, setTrainingData, disposeTrainingData as disposeModelTrainingData, getIsTraining } from "./model.js";
import { setBackend } from "./tf.js";
import { 
  initVizDom, 
  setTrainingDataRef,
  reinitializeModel as vizReinitialize,
  refreshViz as vizRefreshViz,
  updateTrainingStatus as vizUpdateTrainingStatus,
  disposeVizData,
  setStatusMessage,
  setupInputEventListeners
} from "./viz.js";
import { initUiControlsDom, reinitializeModel as uiControlsReinitialize, toggleTraining as uiControlsToggleTraining } from "./ui-controls.js";
import { initPerfectWeightsDom, reinitializeModel as perfectWeightsReinitialize } from "./perfect-weights.js";
import "./orchestrators/reinitializeModel.js";
import "./orchestrators/refreshViz.js";
import "./orchestrators/updateTrainingStatus.js";
import "./orchestrators/toggleTraining.js";

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;

// Define the global orchestrator functions
// These call all module implementations in the correct order

window.reinitializeModel = (newNumLayers: number, newNeuronsPerLayer: number): void => {
  // 1. First, create a new model (model.ts)
  modelReinitialize(newNumLayers, newNeuronsPerLayer);

  // 2. Generate new data (dataset.ts)
  generateData();

  // 3. Set training data in model.ts
  const inputTensor = getTrainingInputTensor();
  const outputTensor = getTrainingOutputTensor();
  if (inputTensor && outputTensor) {
    setTrainingData(inputTensor, outputTensor);
  }

  // 4. Set training data reference for viz lookups
  setTrainingDataRef(getTrainingInputArray(), getTrainingOutputArray());

  // 5. Update visualization (viz.ts)
  vizReinitialize(newNumLayers, newNeuronsPerLayer);

  // 6. Update perfect weights button state (perfect-weights.ts)
  perfectWeightsReinitialize(newNumLayers, newNeuronsPerLayer);

  // 7. Update UI controls state (ui-controls.ts)
  uiControlsReinitialize(newNumLayers, newNeuronsPerLayer);

  // 8. Set ready status
  setStatusMessage('Ready to train!');
};

window.refreshViz = (): void => {
  vizRefreshViz();
};

window.updateTrainingStatus = (epoch: number, loss: number): void => {
  vizUpdateTrainingStatus(epoch, loss);
};

window.toggleTraining = (): void => {
  // 1. Toggle training state in model.ts (start/stop training loop)
  modelToggleTraining();
  
  // 2. Update UI in ui-controls.ts (button text)
  uiControlsToggleTraining();
};

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  if (getIsTraining()) {
    window.toggleTraining();
  }
  disposeTrainingData();
  disposeModelTrainingData();
  disposeVizData();
}

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize module DOM references (each module calls document.getElementById directly)
  initVizDom();
  initUiControlsDom();
  initPerfectWeightsDom();

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
  setupInputEventListeners();

  // Initial setup
  await setBackend(backendSelector.value);
  window.reinitializeModel(numLayers, neuronsPerLayer);
});
