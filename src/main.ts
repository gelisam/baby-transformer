import {
  INPUT_SIZE,
  OUTPUT_SIZE,
  EPOCHS_PER_BATCH,
} from "./constants.js";
import { generateData } from "./dataset.js";
import { createModel } from "./model.js";
import { setBackend } from "./tf.js";
import { VIZ_EXAMPLES_COUNT, pickRandomInputs, updateVizDataFromTextboxes, drawViz, drawLossCurve, drawNetworkArchitecture } from "./viz.js";
import { TrainingData, AppState } from "./types.js";
import { Sequential } from "./tf.js";
import { toggleTrainingMode, updateLayerConfiguration } from "./ui-controls.js";
import { setPerfectWeights, updatePerfectWeightsButton } from "./perfect-weights.js";


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

// Set up backend selection when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
  const trainButton = document.getElementById('train-button') as HTMLButtonElement;
  trainButton.addEventListener('click', () => toggleTrainingMode(appState));
  const perfectWeightsButton = document.getElementById('perfect-weights-button') as HTMLButtonElement;
  perfectWeightsButton.addEventListener('click', () => setPerfectWeights(appState));

  // Add event listener for changes
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  backendSelector.addEventListener('change', async () => {
    // Stop training and clean up old tensors before changing backend
    if (appState.isTraining) {
      toggleTrainingMode(appState); // Toggles isTraining to false
    }
    if (appState.data) {
      appState.data.inputTensor.dispose();
      appState.data.outputTensor.dispose();
    }
    if (appState.vizData) {
      appState.vizData.inputTensor.dispose();
      appState.vizData.outputTensor.dispose();
    }

    await setBackend();
    initializeNewModel(); // Initialize a new model for the new backend
  });

  // Add event listeners for layer configuration sliders
  const numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  const numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  const neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  const neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;

  numLayersSlider.addEventListener('input', () => {
    appState.num_layers = parseInt(numLayersSlider.value, 10);
    numLayersValue.textContent = appState.num_layers.toString();
    updateLayerConfiguration(appState, initializeNewModel, appState.num_layers, appState.neurons_per_layer);
  });

  neuronsPerLayerSlider.addEventListener('input', () => {
    appState.neurons_per_layer = parseInt(neuronsPerLayerSlider.value, 10);
    neuronsPerLayerValue.textContent = appState.neurons_per_layer.toString();
    updateLayerConfiguration(appState, initializeNewModel, appState.num_layers, appState.neurons_per_layer);
  });

  // Add event listeners to the input textboxes
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = document.getElementById(`input-${i}`) as HTMLInputElement;
    if (inputElement) {
      inputElement.addEventListener('input', () => {
        updateVizDataFromTextboxes(appState);
      });
    }
  }

  // Initial setup
  drawNetworkArchitecture(appState);
  await setBackend();
  initializeNewModel();
  updatePerfectWeightsButton(appState);
});

function initializeNewModel(): void {
  // Create a new model
  if (appState.model) {
    appState.model.dispose();
  }
  appState.model = createModel(appState.num_layers, appState.neurons_per_layer);

  // Generate new data
  // No need to clean up old data tensors here, it's handled on backend change
  appState.data = generateData();

  // Generate visualization inputs (only once, not on every frame)
  appState.vizData = pickRandomInputs(appState.data);

  // Reset training state
  appState.currentEpoch = 0;
  appState.lossHistory.length = 0;

  const statusElement = document.getElementById('status')!;
  statusElement.innerHTML = 'Ready to train!';

  // Visualize the initial (untrained) state
  drawViz(appState, appState.vizData);

  // Redraw the architecture in case it changed
  drawNetworkArchitecture(appState);
}

// Visualize the network architecture
