import type { InputFormat } from "../constants.js";
import { setBackend } from "../tf.js";
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
let domInitialized = false;

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;
let inputFormat: InputFormat = 'embedding';

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  // Always stop training (safe to call even if not training)
  window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);
  disposeTrainingData();
  disposeVizData();
}

// Initialize DOM elements by calling document.getElementById directly
function initModelConfigDom() {
  if (domInitialized) return;
  
  backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;
  inputFormatSelector = document.getElementById('input-format-selector') as HTMLSelectElement;
  
  // Set up event listeners
  if (backendSelector) {
    const selector = backendSelector;
    selector.addEventListener('change', async () => {
      prepareForReinitialize();
      await setBackend(selector.value);
      window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
    });
  }
  
  if (numLayersSlider && numLayersValue) {
    const slider = numLayersSlider;
    const valueSpan = numLayersValue;
    slider.addEventListener('input', () => {
      numLayers = parseInt(slider.value, 10);
      valueSpan.textContent = numLayers.toString();
      prepareForReinitialize();
      window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
    });
  }
  
  if (neuronsPerLayerSlider && neuronsPerLayerValue) {
    const slider = neuronsPerLayerSlider;
    const valueSpan = neuronsPerLayerValue;
    slider.addEventListener('input', () => {
      neuronsPerLayer = parseInt(slider.value, 10);
      valueSpan.textContent = neuronsPerLayer.toString();
      prepareForReinitialize();
      window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
    });
  }
  
  if (inputFormatSelector) {
    const selector = inputFormatSelector;
    selector.addEventListener('change', () => {
      inputFormat = selector.value as InputFormat;
      prepareForReinitialize();
      window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
    });
  }
  
  domInitialized = true;
}

// Perform initial setup - set backend and initialize model
async function performInitialSetup(): Promise<void> {
  if (backendSelector) {
    await setBackend(backendSelector.value);
  }
  window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
}

export {
  initModelConfigDom,
  performInitialSetup
};
