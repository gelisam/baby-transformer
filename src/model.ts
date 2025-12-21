import { OUTPUT_SIZE, EPOCHS_PER_BATCH } from "./constants.js";
import { EMBEDDING_DIM, EMBEDDED_INPUT_SIZE, UNEMBEDDING_MATRIX } from "./embeddings.js";
import { tf, Sequential } from "./tf.js";
import { AppState, DomElements } from "./types.js";
import { drawViz, drawLossCurve } from "./viz.js";
import { ReinitializeModelImpl } from "./orchestrators/reinitializeModel.js";

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

async function trainingStep(appState: AppState, dom: DomElements) {
  if (!appState.isTraining) {
    // Training has been paused
    return;
  }

  const statusElement = dom.statusElement;

  // Train for one epoch
  const history = await appState.model.fit(appState.data.inputTensor, appState.data.outputTensor, {
    epochs: EPOCHS_PER_BATCH,
    verbose: 0
  });

  appState.currentEpoch += EPOCHS_PER_BATCH;

  // Get the loss from the last epoch in the batch
  const loss = history.history.loss[history.history.loss.length - 1] as number;
  statusElement.innerHTML = `Training... Epoch ${appState.currentEpoch} - Loss: ${loss.toFixed(4)}`;

  appState.lossHistory.push({ epoch: appState.currentEpoch, loss });
  await drawViz(appState, appState.vizData, dom);
  drawLossCurve(appState, dom);

  // Request the next frame
  requestAnimationFrame(() => trainingStep(appState, dom));
}

// Implementation for the reinitializeModel orchestrator
const reinitializeModel: ReinitializeModelImpl = (appState, dom) => {
  // Create a new model
  if (appState.model) {
    appState.model.dispose();
  }
  appState.model = createModel(appState.num_layers, appState.neurons_per_layer);
};

export { createModel, trainingStep, reinitializeModel };
