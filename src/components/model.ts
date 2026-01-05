import { OUTPUT_SIZE, EPOCHS_PER_BATCH, INPUT_SIZE, EMBEDDING_DIM, getTransformedInputSize } from "../constants.js";
import type { InputFormat } from "../constants.js";
import { EMBEDDING_MATRIX, UNEMBEDDING_MATRIX } from "../embeddings.js";
import { TOKENS } from "../tokens.js";
import { tf, Sequential, Tensor2D } from "../tf.js";
import { Schedule } from "../messageLoop.js";
import { OnEpochCompletedMsg } from "../messages/onEpochCompleted.js";
import { RefreshVizMsg } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetModelWeightsHandler } from "../messages/setModelWeights.js";
import { SetTrainingDataHandler } from "../messages/setTrainingData.js";
import { StartTrainingHandler, StopTrainingHandler } from "../messages/training.js";

let model: Sequential | null = null;
let isTraining = false;
let currentEpoch = 0;
let lossHistory: { epoch: number; loss: number }[] = [];
let trainingInputTensor: Tensor2D | null = null;
let trainingOutputTensor: Tensor2D | null = null;

function createModel(numLayers: number, neuronsPerLayer: number, inputFormat: InputFormat): Sequential {
  const newModel = tf.sequential();
  const vocabSize = TOKENS.length;
  
  if (inputFormat === 'embedding') {
    const embeddingWeights = tf.tensor2d(EMBEDDING_MATRIX);
    newModel.add(tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: EMBEDDING_DIM,
      inputLength: INPUT_SIZE,
      weights: [embeddingWeights],
      trainable: false
    }));
    embeddingWeights.dispose();
    newModel.add(tf.layers.flatten());
  } else if (inputFormat === 'one-hot') {
    const oneHotWeights = tf.eye(vocabSize);
    newModel.add(tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: vocabSize,
      inputLength: INPUT_SIZE,
      weights: [oneHotWeights],
      trainable: false
    }));
    oneHotWeights.dispose();
    newModel.add(tf.layers.flatten());
  }

  const preLayerSize = getTransformedInputSize(inputFormat);

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

  const history = await model.fit(trainingInputTensor, trainingOutputTensor, {
    epochs: EPOCHS_PER_BATCH,
    verbose: 0
  });

  currentEpoch += EPOCHS_PER_BATCH;

  const loss = history.history.loss[history.history.loss.length - 1] as number;
  lossHistory.push({ epoch: currentEpoch, loss });

  window.messageLoop([
    { type: "OnEpochCompleted", epoch: currentEpoch, loss } as OnEpochCompletedMsg,
    { type: "RefreshViz" } as RefreshVizMsg
  ]);

  requestAnimationFrame(() => trainingStep());
}

const reinitializeModel: ReinitializeModelHandler = (_schedule, numLayers, neuronsPerLayer, inputFormat) => {
  if (model) {
    model.dispose();
  }
  model = createModel(numLayers, neuronsPerLayer, inputFormat);
  currentEpoch = 0;
  lossHistory = [];
};

const startTraining: StartTrainingHandler = (_schedule) => {
  isTraining = true;
  requestAnimationFrame(() => trainingStep());
};

const stopTraining: StopTrainingHandler = (_schedule) => {
  isTraining = false;
};

const setTrainingData: SetTrainingDataHandler = (_schedule, data) => {
  trainingInputTensor = data.inputTensor;
  trainingOutputTensor = data.outputTensor;
};

const setModelWeights: SetModelWeightsHandler = (_schedule, weights) => {
  if (model) {
    model.setWeights(weights);
  }
};

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
  setModelWeights,
  getModel,
  getLossHistory,
  getCurrentEpoch,
  disposeTrainingData
};
