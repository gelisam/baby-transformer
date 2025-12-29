import { OUTPUT_SIZE, EPOCHS_PER_BATCH } from "./constants.js";
import { EMBEDDING_DIM, EMBEDDED_INPUT_SIZE, UNEMBEDDING_MATRIX } from "./embeddings.js";
import { tf, Sequential, Tensor2D } from "./tf.js";
import { ReinitializeModel } from "./orchestrators/reinitializeModel.js";
import { StartTraining, StopTraining } from "./orchestrators/training.js";
import { SetTrainingData } from "./orchestrators/setTrainingData.js";
import "./orchestrators/refreshViz.js";
import "./orchestrators/onEpochCompleted.js";

// Module-local state
let model: Sequential | null = null;
let isTraining = false;
let currentEpoch = 0;
let lossHistory: { epoch: number; loss: number }[] = [];
let trainingInputTensor: Tensor2D | null = null;
let trainingOutputTensor: Tensor2D | null = null;

function createModel(numLayers: number, neuronsPerLayer: number): Sequential {
  const newModel = tf.sequential();

  if (numLayers === 0) {
    newModel.add(tf.layers.dense({
      units: EMBEDDING_DIM,
      inputShape: [EMBEDDED_INPUT_SIZE],
      activation: 'linear'
    }));
  } else {
    newModel.add(tf.layers.dense({
      units: neuronsPerLayer,
      inputShape: [EMBEDDED_INPUT_SIZE],
      activation: 'relu'
    }));

    // Remaining hidden layers rely on TensorFlow.js to infer their input shape from the previous layer.
    for (let i = 1; i < numLayers; i++) {
      newModel.add(tf.layers.dense({
        units: neuronsPerLayer,
        activation: 'relu'
      }));
    }

    newModel.add(tf.layers.dense({
      units: EMBEDDING_DIM,
      activation: 'linear'
    }));
  }

  const unembeddingWeights = tf.tensor2d(UNEMBEDDING_MATRIX);
  const unembeddingBias = tf.zeros([OUTPUT_SIZE]);

  newModel.add(tf.layers.dense({
    units: OUTPUT_SIZE,
    activation: 'softmax',
    weights: [unembeddingWeights, unembeddingBias],
    trainable: true
  }));

  unembeddingWeights.dispose();
  unembeddingBias.dispose();

  newModel.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });

  return newModel;
}

async function trainingStep() {
  if (!isTraining || !model || !trainingInputTensor || !trainingOutputTensor) {
    return;
  }

  // Train for one epoch
  const history = await model.fit(trainingInputTensor, trainingOutputTensor, {
    epochs: EPOCHS_PER_BATCH,
    verbose: 0
  });

  currentEpoch += EPOCHS_PER_BATCH;

  // Get the loss from the last epoch in the batch
  const loss = history.history.loss[history.history.loss.length - 1] as number;
  lossHistory.push({ epoch: currentEpoch, loss });

  // Notify other modules via orchestrators
  window.onEpochCompleted(currentEpoch, loss);
  window.refreshViz();

  // Request the next frame
  requestAnimationFrame(() => trainingStep());
}

// Implementation for the reinitializeModel orchestrator
const reinitializeModel: ReinitializeModel = (numLayers, neuronsPerLayer) => {
  // Create a new model
  if (model) {
    model.dispose();
  }
  model = createModel(numLayers, neuronsPerLayer);
  
  // Reset training state
  currentEpoch = 0;
  lossHistory = [];
};

// Implementation for the startTraining orchestrator
const startTraining: StartTraining = () => {
  isTraining = true;
  requestAnimationFrame(() => trainingStep());
};

// Implementation for the stopTraining orchestrator
const stopTraining: StopTraining = () => {
  isTraining = false;
};

// Implementation for the setTrainingData orchestrator
const setTrainingData: SetTrainingData = (data) => {
  trainingInputTensor = data.inputTensor;
  trainingOutputTensor = data.outputTensor;
};

// Getters for external access
function getModel(): Sequential | null {
  return model;
}

function getLossHistory(): { epoch: number; loss: number }[] {
  return lossHistory;
}

function getCurrentEpoch(): number {
  return currentEpoch;
}

function disposeTrainingData() {
  if (trainingInputTensor) {
    try { trainingInputTensor.dispose(); } catch (e) { /* ignore */ }
    trainingInputTensor = null;
  }
  if (trainingOutputTensor) {
    try { trainingOutputTensor.dispose(); } catch (e) { /* ignore */ }
    trainingOutputTensor = null;
  }
}

export { 
  createModel, 
  reinitializeModel, 
  startTraining,
  stopTraining,
  setTrainingData,
  getModel, 
  getLossHistory,
  getCurrentEpoch,
  disposeTrainingData
};
