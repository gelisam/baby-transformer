import { EPOCHS_PER_BATCH } from "../constants.js";
import { INPUT_SIZE, EMBEDDING_DIM, getOutputSize, getTransformedInputSize } from "../tokens.js";
import type { InputFormat } from "../tokens.js";
import { EMBEDDING_MATRIX, UNEMBEDDING_MATRIX } from "../embeddings.js";
import { tf, Sequential, Tensor2D } from "../tf.js";
import { Schedule } from "../messageLoop.js";
import { OnEpochCompletedMsg } from "../messages/onEpochCompleted.js";
import { RefreshVizMsg } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetModelWeightsHandler } from "../messages/setModelWeights.js";
import { SetTrainingDataHandler } from "../messages/setTrainingData.js";
import { StartTrainingHandler, StopTrainingHandler } from "../messages/training.js";

// Module-local state
let model: Sequential | null = null;
let isTraining = false;
let currentEpoch = 0;
let lossHistory: { epoch: number; loss: number }[] = [];
let trainingInputTensor: Tensor2D | null = null;
let trainingOutputTensor: Tensor2D | null = null;

function createModel(numLayers: number, neuronsPerLayer: number, inputFormat: InputFormat, vocabSize: number): Sequential {
  const newModel = tf.sequential();
  const outputSize = getOutputSize(vocabSize);
  
  // Add preprocessing layer based on input format
  // Input is always [batchSize, INPUT_SIZE] with token indices (0-5)
  if (inputFormat === 'embedding') {
    // Embedding layer converts token indices to embedding vectors
    // Then flatten to get [batchSize, INPUT_SIZE * EMBEDDING_DIM]
    const embeddingWeights = tf.tensor2d(EMBEDDING_MATRIX); // [vocabSize, EMBEDDING_DIM]
    newModel.add(tf.layers.embedding({
      inputDim: vocabSize * 2,
      outputDim: EMBEDDING_DIM,
      inputLength: INPUT_SIZE,
      weights: [embeddingWeights],
      trainable: false
    }));
    embeddingWeights.dispose();
    newModel.add(tf.layers.flatten());
  } else if (inputFormat === 'one-hot') {
    // One-hot encoding via embedding layer with identity matrix
    // Then flatten to get [batchSize, INPUT_SIZE * vocabSize]
    const oneHotWeights = tf.eye(vocabSize * 2);
    newModel.add(tf.layers.embedding({
      inputDim: vocabSize * 2,
      outputDim: vocabSize * 2,
      inputLength: INPUT_SIZE,
      weights: [oneHotWeights],
      trainable: false
    }));
    oneHotWeights.dispose();
    newModel.add(tf.layers.flatten());
  }
  // For 'number' format, no preprocessing layer - input is used directly
  
  const preLayerSize = getTransformedInputSize(inputFormat, vocabSize);

  if (numLayers === 0) {
    newModel.add(tf.layers.dense({
      units: EMBEDDING_DIM,
      inputShape: inputFormat === 'number' ? [preLayerSize] : undefined,
      activation: 'linear'
    }));
  } else {
    newModel.add(tf.layers.dense({
      units: neuronsPerLayer,
      inputShape: inputFormat === 'number' ? [preLayerSize] : undefined,
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
  const unembeddingBias = tf.zeros([outputSize]);

  newModel.add(tf.layers.dense({
    units: outputSize,
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

  // Notify other modules via messages
  window.messageLoop([
    { type: "OnEpochCompleted", epoch: currentEpoch, loss } as OnEpochCompletedMsg,
    { type: "RefreshViz" } as RefreshVizMsg
  ]);

  // Request the next frame
  requestAnimationFrame(() => trainingStep());
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, numLayers, neuronsPerLayer, inputFormat, vocabSize) => {
  // Create a new model
  if (model) {
    model.dispose();
  }
  model = createModel(numLayers, neuronsPerLayer, inputFormat, vocabSize);
  
  // Reset training state
  currentEpoch = 0;
  lossHistory = [];
};

// Implementation for the startTraining message handler
const startTraining: StartTrainingHandler = (_schedule) => {
  isTraining = true;
  requestAnimationFrame(() => trainingStep());
};

// Implementation for the stopTraining message handler
const stopTraining: StopTrainingHandler = (_schedule) => {
  isTraining = false;
};

// Implementation for the setTrainingData message handler
const setTrainingData: SetTrainingDataHandler = (_schedule, data) => {
  trainingInputTensor = data.inputTensor;
  trainingOutputTensor = data.outputTensor;
};

// Implementation for the setModelWeights message handler
const setModelWeights: SetModelWeightsHandler = (_schedule, weights) => {
  if (model) {
    model.setWeights(weights);
  }
};

// Getter for model - used by viz.ts for predictions
function getModel(): Sequential | null {
  return model;
}

// Getters for external access
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
  setModelWeights,
  getModel,
  getLossHistory,
  getCurrentEpoch,
  disposeTrainingData
};
