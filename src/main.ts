import { generateData } from "./dataset.js";
import { reinitializeModel as modelReinitialize } from "./model.js";
import { setBackend } from "./tf.js";
import { VIZ_EXAMPLES_COUNT, updateVizDataFromTextboxes, reinitializeModel as vizReinitialize } from "./viz.js";
import { TrainingData, AppState, DomElements } from "./types.js";
import { Sequential } from "./tf.js";
import { toggleTrainingMode, reinitializeModel as uiControlsReinitialize } from "./ui-controls.js";
import { setPerfectWeights, reinitializeModel as perfectWeightsReinitialize } from "./perfect-weights.js";
import "./orchestrators/reinitializeModel.js";

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

// Define the global orchestrator function
// This calls all module implementations in the correct order
window.reinitializeModel = (appState: AppState, dom: DomElements): void => {
  // 1. First, create a new model (model.ts)
  modelReinitialize(appState, dom);

  // 2. Generate new data (dataset.ts - called inline as it's simple)
  appState.data = generateData();

  // 3. Reset training state
  appState.currentEpoch = 0;
  appState.lossHistory.length = 0;
  dom.statusElement.innerHTML = 'Ready to train!';

  // 4. Update visualization (viz.ts)
  vizReinitialize(appState, dom);

  // 5. Update perfect weights button state (perfect-weights.ts)
  perfectWeightsReinitialize(appState, dom);

  // 6. Update UI controls state (ui-controls.ts)
  uiControlsReinitialize(appState, dom);
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
    if (appState.data) {
      appState.data.inputTensor.dispose();
      appState.data.outputTensor.dispose();
    }
    if (appState.vizData) {
      appState.vizData.inputTensor.dispose();
      appState.vizData.outputTensor.dispose();
    }

    await setBackend(dom.backendSelector.value);
    window.reinitializeModel(appState, dom);
  });

  // Add event listeners for layer configuration sliders
  dom.numLayersSlider.addEventListener('input', () => {
    appState.num_layers = parseInt(dom.numLayersSlider.value, 10);
    dom.numLayersValue.textContent = appState.num_layers.toString();
    // Stop training before reinitializing
    if (appState.isTraining) {
      toggleTrainingMode(appState, dom);
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
    window.reinitializeModel(appState, dom);
  });

  dom.neuronsPerLayerSlider.addEventListener('input', () => {
    appState.neurons_per_layer = parseInt(dom.neuronsPerLayerSlider.value, 10);
    dom.neuronsPerLayerValue.textContent = appState.neurons_per_layer.toString();
    // Stop training before reinitializing
    if (appState.isTraining) {
      toggleTrainingMode(appState, dom);
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
    window.reinitializeModel(appState, dom);
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

  // Initial setup - draw network architecture before model init
  vizReinitialize(appState, dom); // This draws the network architecture
  await setBackend(dom.backendSelector.value);
  window.reinitializeModel(appState, dom);
});
