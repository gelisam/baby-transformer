import { setBackend } from "./tf.js";
import { VIZ_EXAMPLES_COUNT, updateVizDataFromTextboxes, drawNetworkArchitecture } from "./viz.js";
import { TrainingData, AppState, DomElements } from "./types.js";
import { Sequential } from "./tf.js";
import { toggleTrainingMode, updateLayerConfiguration } from "./ui-controls.js";
import { setPerfectWeights, updatePerfectWeightsButton } from "./perfect-weights.js";
import { bootstrapAppState, disposeAppState, resetAppState } from "./app-lifecycle.js";

const appState: AppState = {
  model: undefined as unknown as Sequential,
  isTraining: false,
  currentEpoch: 0,
  lossHistory: [] as { epoch: number, loss: number }[],
  data: undefined as unknown as TrainingData,
  vizData: undefined as unknown as TrainingData,
  num_layers: 4,
  neurons_per_layer: 6
};

document.addEventListener('DOMContentLoaded', async () => {
  const dom: DomElements = {
    trainButton: document.getElementById('train-button') as HTMLButtonElement,
    perfectWeightsButton: document.getElementById('perfect-weights-button') as HTMLButtonElement,
    perfectWeightsTooltipText: document.getElementById('perfect-weights-tooltip-text') as HTMLSpanElement,
    backendSelector: document.getElementById('backend-selector') as HTMLSelectElement,
    numLayersSlider: document.getElementById('num-layers-slider') as HTMLInputElement,
    numLayersValue: document.getElementById('num-layers-value') as HTMLSpanElement,
    neuronsPerLayerSlider: document.getElementById('neurons-per-layer-slider') as HTMLInputElement,
    neuronsPerLayerValue: document.getElementById('neurons-per-layer-value') as HTMLSpanElement,
    inputElements: Array.from({ length: VIZ_EXAMPLES_COUNT }, (_, i) => document.getElementById(`input-${i}`) as HTMLInputElement),
    statusElement: document.getElementById('status')!,
    outputCanvas: document.getElementById('output-canvas') as HTMLCanvasElement,
    lossCanvas: document.getElementById('loss-canvas') as HTMLCanvasElement,
    networkCanvas: document.getElementById('network-canvas') as HTMLCanvasElement,
    toaster: document.getElementById('toaster') as HTMLElement
  };

  dom.trainButton.addEventListener('click', () => toggleTrainingMode(appState, dom));
  dom.perfectWeightsButton.addEventListener('click', () => setPerfectWeights(appState, dom));

  // Add event listener for changes
  dom.backendSelector.addEventListener('change', async () => {
    // Stop training and clean up old tensors before changing backend
    if (appState.isTraining) {
      toggleTrainingMode(appState, dom); // Toggles isTraining to false
    }

    disposeAppState(appState);
    await setBackend(dom.backendSelector);
    bootstrapAppState(appState, dom); // Initialize a new model for the new backend
    updatePerfectWeightsButton(appState, dom);
  });

  // Add event listeners for layer configuration sliders
  dom.numLayersSlider.addEventListener('input', () => {
    appState.num_layers = parseInt(dom.numLayersSlider.value, 10);
    dom.numLayersValue.textContent = appState.num_layers.toString();
    updateLayerConfiguration(appState, dom, () => initializeApp(dom));
  });

  dom.neuronsPerLayerSlider.addEventListener('input', () => {
    appState.neurons_per_layer = parseInt(dom.neuronsPerLayerSlider.value, 10);
    dom.neuronsPerLayerValue.textContent = appState.neurons_per_layer.toString();
    updateLayerConfiguration(appState, dom, () => initializeApp(dom));
  });

  // Add event listeners to the input textboxes
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = dom.inputElements[i];
    if (inputElement) {
      inputElement.addEventListener('input', () => {
        updateVizDataFromTextboxes(appState, dom);
      });
    }
  }

  // Initial setup
  drawNetworkArchitecture(appState, dom);
  await setBackend(dom.backendSelector);
  initializeApp(dom);
});

function initializeApp(dom: DomElements): void {
  resetAppState(appState, dom);
  updatePerfectWeightsButton(appState, dom);
}
