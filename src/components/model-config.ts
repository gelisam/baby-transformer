import type { InputFormat } from "../inputFormat.js";
import { setBackend } from "../tf.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelMsg } from "../messages/reinitializeModel.js";
import { StopTrainingMsg } from "../messages/training.js";
import { disposeTrainingData } from "./model.js";
import { disposeVizData } from "./viz-examples.js";

// Module-local state for DOM elements (initialized on first use)
let backendSelector: HTMLSelectElement | null = null;
let vocabSizeSlider: HTMLInputElement | null = null;
let vocabSizeSpan: HTMLSpanElement | null = null;
let numLayersSlider: HTMLInputElement | null = null;
let numLayersSpan: HTMLSpanElement | null = null;
let neuronsPerLayerSlider: HTMLInputElement | null = null;
let neuronsPerLayerSpan: HTMLSpanElement | null = null;
let inputFormatSelector: HTMLSelectElement | null = null;

// Module-local state for layer configuration
let vocabSize = 3;
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

function getVocabSizeSlider(): HTMLInputElement {
  if (!vocabSizeSlider) {
    vocabSizeSlider = document.getElementById('vocab-size-slider') as HTMLInputElement;
  }
  return vocabSizeSlider;
}

function getVocabSizeSpan(): HTMLSpanElement {
  if (!vocabSizeSpan) {
    vocabSizeSpan = document.getElementById('vocab-size-span') as HTMLSpanElement;
  }
  return vocabSizeSpan;
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
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
  });
  
  const vocabSlider = getVocabSizeSlider();
  const vocabValueSpan = getVocabSizeSpan();
  vocabSlider.addEventListener('input', () => {
    vocabSize = parseInt(vocabSlider.value, 10);
    vocabValueSpan.textContent = vocabSize.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
  });
  
  const layersSlider = getNumLayersSlider();
  const layersValueSpan = getNumLayersSpan();
  layersSlider.addEventListener('input', () => {
    numLayers = parseInt(layersSlider.value, 10);
    layersValueSpan.textContent = numLayers.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
  });
  
  const neuronsSlider = getNeuronsPerLayerSlider();
  const neuronsValueSpan = getNeuronsPerLayerSpan();
  neuronsSlider.addEventListener('input', () => {
    neuronsPerLayer = parseInt(neuronsSlider.value, 10);
    neuronsValueSpan.textContent = neuronsPerLayer.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
  });
  
  const formatSelector = getInputFormatSelector();
  formatSelector.addEventListener('change', () => {
    inputFormat = formatSelector.value as InputFormat;
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
  });
};

// Perform initial setup - set backend and initialize model
async function performInitialSetup(): Promise<void> {
  const selector = getBackendSelector();
  await setBackend(selector.value);
  window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat, vocabSize } as ReinitializeModelMsg);
}

export {
  init,
  performInitialSetup
};
