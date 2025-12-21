import { generateData, getTrainingInputArray, getTrainingOutputArray, getTrainingInputTensor, getTrainingOutputTensor, disposeTrainingData } from "./dataset.js";
import { reinitializeModel as modelReinitialize, setTrainingData, disposeTrainingData as disposeModelTrainingData, getIsTraining } from "./model.js";
import { setBackend } from "./tf.js";
import { 
  VIZ_EXAMPLES_COUNT, 
  initVizDom, 
  setTrainingDataRef,
  updateVizDataFromTextboxes, 
  reinitializeModel as vizReinitialize,
  refreshViz as vizRefreshViz,
  updateTrainingStatus as vizUpdateTrainingStatus,
  disposeVizData,
  setStatusMessage
} from "./viz.js";
import { initUiControlsDom, toggleTrainingMode, reinitializeModel as uiControlsReinitialize } from "./ui-controls.js";
import { initPerfectWeightsDom, setPerfectWeights, reinitializeModel as perfectWeightsReinitialize } from "./perfect-weights.js";
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
  toggleTrainingMode();
};

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  if (getIsTraining()) {
    toggleTrainingMode();
  }
  disposeTrainingData();
  disposeModelTrainingData();
  disposeVizData();
}

document.addEventListener('DOMContentLoaded', async () => {
  // Get DOM elements
  const trainButton = document.getElementById('train-button') as HTMLButtonElement;
  const perfectWeightsButton = document.getElementById('perfect-weights-button') as HTMLButtonElement;
  const perfectWeightsTooltipText = document.getElementById('perfect-weights-tooltip-text') as HTMLSpanElement;
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  const numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  const numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  const neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  const neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;
  const inputElements = Array.from({ length: VIZ_EXAMPLES_COUNT }, (_, i) => document.getElementById(`input-${i}`) as HTMLInputElement);
  const statusElement = document.getElementById('status')!;
  const outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  const lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  const networkCanvas = document.getElementById('network-canvas') as HTMLCanvasElement;

  // Initialize module DOM references
  initVizDom(outputCanvas, lossCanvas, networkCanvas, inputElements, statusElement);
  initUiControlsDom(trainButton);
  initPerfectWeightsDom(perfectWeightsButton, perfectWeightsTooltipText);

  // Set up event listeners
  trainButton.addEventListener('click', () => window.toggleTraining());
  perfectWeightsButton.addEventListener('click', () => setPerfectWeights());

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

  // Add event listeners to the input textboxes
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = inputElements[i];
    if (inputElement) {
      inputElement.addEventListener('input', () => {
        updateVizDataFromTextboxes();
      });
    }
  }

  // Initial setup
  await setBackend(backendSelector.value);
  window.reinitializeModel(numLayers, neuronsPerLayer);
});
