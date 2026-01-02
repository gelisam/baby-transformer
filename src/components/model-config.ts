import type { InputFormat } from "../constants.js";
import { setBackend } from "../tf.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelMsg } from "../messages/reinitializeModel.js";
import { StopTrainingMsg } from "../messages/training.js";
import { disposeTrainingData } from "./model.js";
import { disposeVizData } from "./viz-examples.js";

// Module-local state for DOM elements (initialized on first use)
let backendSelector: HTMLSelectElement | null = null;
let numLayersSlider: HTMLInputElement | null = null;
let numLayersValue: HTMLSpanElement | null = null;
let neuronsPerLayerSlider: HTMLInputElement | null = null;
let neuronsPerLayerValue: HTMLSpanElement | null = null;
let inputFormatSelector: HTMLSelectElement | null = null;

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;
let inputFormat: InputFormat = 'embedding';

// Getter functions that check and initialize DOM elements if needed
function getBackendSelector(): HTMLSelectElement {
  if (!backendSelector) {
    backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  }
  return backendSelector;
}

function getNumLayersSlider(): HTMLInputElement {
  if (!numLayersSlider) {
    numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  }
  return numLayersSlider;
}

function getNumLayersValue(): HTMLSpanElement {
  if (!numLayersValue) {
    numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  }
  return numLayersValue;
}

function getNeuronsPerLayerSlider(): HTMLInputElement {
  if (!neuronsPerLayerSlider) {
    neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  }
  return neuronsPerLayerSlider;
}

function getNeuronsPerLayerValue(): HTMLSpanElement {
  if (!neuronsPerLayerValue) {
    neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;
  }
  return neuronsPerLayerValue;
}

function getInputFormatSelector(): HTMLSelectElement {
  if (!inputFormatSelector) {
    inputFormatSelector = document.getElementById('input-format-selector') as HTMLSelectElement;
  }
  return inputFormatSelector;
}

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  // Always stop training (safe to call even if not training)
  window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);
  disposeTrainingData();
  disposeVizData();
}

// Handler for the Init message - attach event listeners
const init: InitHandler = (_schedule) => {
  const selector = getBackendSelector();
  selector.addEventListener('change', async () => {
    prepareForReinitialize();
    await setBackend(selector.value);
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });
  
  const layersSlider = getNumLayersSlider();
  const layersValueSpan = getNumLayersValue();
  layersSlider.addEventListener('input', () => {
    numLayers = parseInt(layersSlider.value, 10);
    layersValueSpan.textContent = numLayers.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });
  
  const neuronsSlider = getNeuronsPerLayerSlider();
  const neuronsValueSpan = getNeuronsPerLayerValue();
  neuronsSlider.addEventListener('input', () => {
    neuronsPerLayer = parseInt(neuronsSlider.value, 10);
    neuronsValueSpan.textContent = neuronsPerLayer.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });
  
  const formatSelector = getInputFormatSelector();
  formatSelector.addEventListener('change', () => {
    inputFormat = formatSelector.value as InputFormat;
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });
};

// Perform initial setup - set backend and initialize model
async function performInitialSetup(): Promise<void> {
  const selector = getBackendSelector();
  await setBackend(selector.value);
  window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
}

export {
  init,
  performInitialSetup
};
