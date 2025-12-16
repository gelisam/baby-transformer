import { EMBEDDING_DIM, EMBEDDED_INPUT_SIZE, OUTPUT_SIZE } from "./constants.js";
import { UNEMBEDDING_MATRIX } from "./embeddings.js";
import { tf, Sequential } from "./tf.js";

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

    for (let i = 1; i < numLayers; i++) {
      model.add(tf.layers.dense({
        units: neuronsPerLayer,
        inputShape: [EMBEDDED_INPUT_SIZE],
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

export { createModel };
