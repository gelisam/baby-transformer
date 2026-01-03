import type { InputFormat, OutputFormat } from "../constants.js";
import { setBackend } from "../tf.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelMsg } from "../messages/reinitializeModel.js";
import { StopTrainingMsg } from "../messages/training.js";
import { disposeTrainingData } from "./model.js";
import { disposeVizData } from "./viz-examples.js";

// Module-local state for DOM elements (initialized on first use)
let backendSelector: HTMLSelectElement | null = null;
let numLayersSlider: HTMLInputElement | null = null;
let numLayersSpan: HTMLSpanElement | null = null;
let neuronsPerLayerSlider: HTMLInputElement | null = null;
let neuronsPerLayerSpan: HTMLSpanElement | null = null;
let inputFormatSelector: HTMLSelectElement | null = null;
let outputFormatSelector: HTMLSelectElement | null = null;

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;
let inputFormat: InputFormat = 'embedding';
let outputFormat: OutputFormat = 'probabilities';

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

function getNumLayersSpan(): HTMLSpanElement {
  if (!numLayersSpan) {
    numLayersSpan = document.getElementById('num-layers-span') as HTMLSpanElement;
  }
  return numLayersSpan;
}

function getNeuronsPerLayerSlider(): HTMLInputElement {
  if (!neuronsPerLayerSlider) {
    neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  }
  return neuronsPerLayerSlider;
}

function getNeuronsPerLayerSpan(): HTMLSpanElement {
  if (!neuronsPerLayerSpan) {
    neuronsPerLayerSpan = document.getElementById('neurons-per-layer-span') as HTMLSpanElement;
  }
  return neuronsPerLayerSpan;
}

function getInputFormatSelector(): HTMLSelectElement {
  if (!inputFormatSelector) {
    inputFormatSelector = document.getElementById('input-format-selector') as HTMLSelectElement;
  }
  return inputFormatSelector;
}

function getOutputFormatSelector(): HTMLSelectElement {
  if (!outputFormatSelector) {
    outputFormatSelector = document.getElementById('output-format-selector') as HTMLSelectElement;
  }
  return outputFormatSelector;
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
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
  });
  
  const layersSlider = getNumLayersSlider();
  const layersValueSpan = getNumLayersSpan();
  layersSlider.addEventListener('input', () => {
    numLayers = parseInt(layersSlider.value, 10);
    layersValueSpan.textContent = numLayers.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
  });
  
  const neuronsSlider = getNeuronsPerLayerSlider();
  const neuronsValueSpan = getNeuronsPerLayerSpan();
  neuronsSlider.addEventListener('input', () => {
    neuronsPerLayer = parseInt(neuronsSlider.value, 10);
    neuronsValueSpan.textContent = neuronsPerLayer.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
  });
  
  const formatSelector = getInputFormatSelector();
  formatSelector.addEventListener('change', () => {
    inputFormat = formatSelector.value as InputFormat;
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
  });
  
  const outFormatSelector = getOutputFormatSelector();
  outFormatSelector.addEventListener('change', () => {
    outputFormat = outFormatSelector.value as OutputFormat;
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
  });
};

// Perform initial setup - set backend and initialize model
async function performInitialSetup(): Promise<void> {
  const selector = getBackendSelector();
  await setBackend(selector.value);
  window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, outputFormat } as ReinitializeModelMsg);
}

export {
  init,
  performInitialSetup
};
