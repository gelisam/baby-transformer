/**
 * Neural network model creation and training.
 * 
 * This module handles the creation of the transformer-like model and
 * manages the training loop with visualization updates.
 */

import { OUTPUT_SIZE } from "./constants.js";
import { EMBEDDING_DIM, EMBEDDED_INPUT_SIZE, UNEMBEDDING_MATRIX } from "./embeddings.js";
import { tf, Sequential } from "./tf.js";
import { AppState, DomElements } from "./types.js";
import { drawViz, drawLossCurve } from "./viz.js";
import { TRAINING_CONFIG, UI_MESSAGES } from "./config.js";

/**
 * Create a sequential neural network model.
 * 
 * The model architecture consists of:
 * - Embedding layer (converts tokens to embeddings)
 * - Hidden ReLU layers (configurable number and size)
 * - Linear layer (reduces to embedding dimension)
 * - Unembedding + softmax layer (produces token probabilities)
 * 
 * @param numLayers - Number of hidden layers
 * @param neuronsPerLayer - Number of neurons in each hidden layer
 * @returns A compiled TensorFlow.js Sequential model
 */
function createModel(numLayers: number, neuronsPerLayer: number): Sequential {
  const model = tf.sequential();

  if (numLayers === 0) {
    model.add(tf.layers.dense({
      units: EMBEDDING_DIM,
      inputShape: [EMBEDDED_INPUT_SIZE],
      activation: 'linear'
    }));
  } else {
    model.add(tf.layers.dense({
      units: neuronsPerLayer,
      inputShape: [EMBEDDED_INPUT_SIZE],
      activation: 'relu'
    }));

    // Remaining hidden layers rely on TensorFlow.js to infer their input shape from the previous layer.
    for (let i = 1; i < numLayers; i++) {
      model.add(tf.layers.dense({
        units: neuronsPerLayer,
        activation: 'relu'
      }));
    }

    model.add(tf.layers.dense({
      units: EMBEDDING_DIM,
      activation: 'linear'
    }));
  }

  const unembeddingWeights = tf.tensor2d(UNEMBEDDING_MATRIX);
  const unembeddingBias = tf.zeros([OUTPUT_SIZE]);

  model.add(tf.layers.dense({
    units: OUTPUT_SIZE,
    activation: 'softmax',
    weights: [unembeddingWeights, unembeddingBias],
    trainable: true
  }));

  unembeddingWeights.dispose();
  unembeddingBias.dispose();

  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });

  return model;
}

/**
 * Execute a single training step.
 * 
 * This function trains the model for one batch of epochs, updates the UI
 * with current loss and visualizations, and schedules the next training step
 * if training is still active.
 * 
 * @param appState - The application state containing the model and training data
 * @param dom - DOM elements for UI updates
 */
async function trainingStep(appState: AppState, dom: DomElements) {
  if (!appState.isTraining) {
    // Training has been paused
    return;
  }

  const statusElement = dom.statusElement;

  // Train for one epoch
  const history = await appState.model.fit(appState.data.inputTensor, appState.data.outputTensor, {
    epochs: TRAINING_CONFIG.epochsPerBatch,
    verbose: 0
  });

  appState.currentEpoch += TRAINING_CONFIG.epochsPerBatch;

  // Get the loss from the last epoch in the batch
  const loss = history.history.loss[history.history.loss.length - 1] as number;
  statusElement.innerHTML = UI_MESSAGES.training(appState.currentEpoch, loss);

  appState.lossHistory.push({ epoch: appState.currentEpoch, loss });
  await drawViz(appState, appState.vizData, dom);
  drawLossCurve(appState, dom);

  // Request the next frame
  requestAnimationFrame(() => trainingStep(appState, dom));
}

export { createModel, trainingStep };
