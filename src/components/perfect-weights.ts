import { INPUT_SIZE } from "../inputFormat.js";
import { getOutputSize } from "../tokens.js";
import { tf, Tensor } from "../tf.js";
import { Schedule } from "../messageLoop.js";
import { InitHandler } from "../messages/init.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetModelWeightsMsg } from "../messages/setModelWeights.js";
import { StopTrainingMsg } from "../messages/training.js";
import { drawViz } from "./viz-examples.js";

// Module-local state for DOM elements (initialized on first use)
let perfectWeightsButton: HTMLButtonElement | null = null;
let perfectWeightsTooltipText: HTMLSpanElement | null = null;

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;
let currentVocabSize = 3;

// Getter functions that check and initialize DOM elements if needed
function getPerfectWeightsButton(): HTMLButtonElement {
  if (!perfectWeightsButton) {
    perfectWeightsButton = document.getElementById('perfect-weights-button') as HTMLButtonElement;
  }
  return perfectWeightsButton;
}

function getPerfectWeightsTooltipText(): HTMLSpanElement {
  if (!perfectWeightsTooltipText) {
    perfectWeightsTooltipText = document.getElementById('perfect-weights-tooltip-text') as HTMLSpanElement;
  }
  return perfectWeightsTooltipText;
}

// Forward declaration for event handler
async function handlePerfectWeightsClick(): Promise<void> {
  await setPerfectWeights();
}

// Handler for the Init message - attach event listeners
const init: InitHandler = (_schedule) => {
  const button = getPerfectWeightsButton();
  button.addEventListener('click', () => handlePerfectWeightsClick());
};

function canUsePerfectWeights(layers: number, neurons: number): { canUse: boolean, reason: string } {
  // The setPerfectWeights function needs to be updated for the new embedding/unembedding architecture
  return {
    canUse: false,
    reason: 'Perfect weights feature not yet implemented for embedding/unembedding architecture.'
  };
}

function updatePerfectWeightsButton(): void {
  const button = getPerfectWeightsButton();
  const tooltipText = getPerfectWeightsTooltipText();
  
  const result = canUsePerfectWeights(numLayers, neuronsPerLayer);

  button.disabled = !result.canUse;

  if (!result.canUse) {
    tooltipText.textContent = result.reason;
  } else {
    tooltipText.textContent = '';
  }
}

async function setPerfectWeights(): Promise<void> {
  // Always stop training when setting perfect weights
  window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);

  // We need to complete this:
  //   <letter1>=<number1> <letter2>=<number2> <letter3>=____
  //
  // Using this algorithm:
  //   if (letter3 == letter1) {
  //     oneHot(number1)
  //   } else if (letter3 == letter2) {
  //     oneHot(number2)
  //   } else {
  //     <don't care>
  //   }
  //
  // Or equivalently:
  //   oneHot(
  //     valueIfEqual(number1, letter3, letter1) +
  //     valueIfEqual(number2, letter3, letter2)
  //   )
  //   where
  //     oneHot(v) = [
  //       isEqual(v, 1),
  //       isEqual(v, 2),
  //       isEqual(v, 3)
  //     ]
  //     valueIfEqual(v, x, y) = relu(v - 1000 * notEqual(x, y))
  //     notEqual(x, y) = relu(x - y) + relu(y - x)
  //     isEqual(x, v) = relu(1 - relu(x - v) - relu(v - x))
  //
  // Inlining everything:
  //
  //   const not1 = relu(letter3 - letter1) + relu(letter1 - letter3)
  //   const not2 = relu(letter3 - letter2) + relu(letter2 - letter3)
  //   const contribution1 = relu(number1 - 1000 * not1)
  //   const contribution2 = relu(number2 - 1000 * not2)
  //   const output = contribution1 + contribution2
  //   const sub1fromOut = relu(output - 1)
  //   const sub2fromOut = relu(output - 2)
  //   const sub3fromOut = relu(output - 3)
  //   const subOutFrom1 = relu(1 - output)
  //   const subOutFrom2 = relu(2 - output)
  //   const subOutFrom3 = relu(3 - output)
  //   [
  //     relu(1 - sub1FromOut - subOutFrom1),
  //     relu(1 - sub2FromOut - subOutFrom2),
  //     relu(1 - sub3FromOut - subOutFrom3)
  //   ]
  //
  // Simplifying:
  //
  //   const sub1from3 = relu(letter3 - letter1)
  //   const sub2from3 = relu(letter3 - letter2)
  //   const sub3from1 = relu(letter1 - letter3)
  //   const sub3from2 = relu(letter2 - letter3)
  //   const contribution1 = relu(number1 - 1000 * sub1from3 - 1000 * sub3from1)
  //   const contribution2 = relu(number2 - 1000 * sub2from3 - 1000 * sub3from2)
  //   const sub1fromOut = relu(contribution1 + contribution2 - 1)
  //   const sub2fromOut = relu(contribution1 + contribution2 - 2)
  //   const sub3fromOut = relu(contribution1 + contribution2 - 3)
  //   const subOutFrom1 = relu(1 - contribution1 - contribution2)
  //   const subOutFrom2 = relu(2 - contribution1 - contribution2)
  //   const subOutFrom3 = relu(3 - contribution1 - contribution2)
  //   [
  //     relu(1 - sub1FromOut - subOutFrom1),
  //     relu(1 - sub2FromOut - subOutFrom2),
  //     relu(1 - sub3FromOut - subOutFrom3),
  //     0,
  //     0,
  //     0
  //   ]
  //
  // Spelling out the weights and layers:
  //
  //   // hidden layer 1
  //   const sub1from3 = relu(1.0 * letter3 + -1.0 * letter1)
  //   const sub3from1 = relu(1.0 * letter1 + -1.0 * letter3)
  //   const sub2from3 = relu(1.0 * letter3 + -1.0 * letter2)
  //   const sub3from2 = relu(1.0 * letter2 + -1.0 * letter3)
  //   const number1layer1 = relu(1.0 * number1)
  //   const number2layer1 = relu(1.0 * number2)
  //
  //   // hidden layer 2
  //   const contribution1 = relu(1.0 * number1layer1 + -1000.0 * sub1from3 + -1000.0 * sub3from1)
  //   const contribution2 = relu(1.0 * number2layer1 + -1000.0 * sub2from3 + -1000.0 * sub3from2)
  //
  //   // hidden layer 3
  //   const sub1fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 1.0)
  //   const sub2fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 2.0)
  //   const sub3fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 3.0)
  //   const subOutFrom1 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 1.0)
  //   const subOutFrom2 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 2.0)
  //   const subOutFrom3 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 3.0)
  //
  //   // hidden layer 4
  //   [
  //     relu(-1.0 * sub1FromOut + -1.0 * subOutFrom1 + 1.0),
  //     relu(-1.0 * sub2FromOut + -1.0 * subOutFrom2 + 1.0),
  //     relu(-1.0 * sub3FromOut + -1.0 * subOutFrom3 + 1.0),
  //     relu(0.0),
  //     relu(0.0),
  //     relu(0.0)
  //   ]

  // Name the index of the neuron implementing the corresponding variable
  const letter1 = 0;
  const number1 = 1;
  const letter2 = 2;
  const number2 = 3;
  const letter3 = 4;

  const /*mut*/ layer1weights = tf.buffer([INPUT_SIZE, neuronsPerLayer])
  const /*mut*/ layer1bias = tf.buffer([neuronsPerLayer]);
  const sub1from3 = 0;
  const sub3from1 = 1;
  const sub2from3 = 2;
  const sub3from2 = 3;
  const number1layer1 = 4;
  const number2layer1 = 5;
  // const sub1from3 = relu(1.0 * letter3 + -1.0 * letter1)
  layer1weights.set(1.0, letter3, sub1from3);
  layer1weights.set(-1.0, letter1, sub1from3);
  // const sub3from1 = relu(1.0 * letter1 + -1.0 * letter3)
  layer1weights.set(1.0, letter1, sub3from1);
  layer1weights.set(-1.0, letter3, sub3from1);
  // const sub2from3 = relu(1.0 * letter3 + -1.0 * letter2)
  layer1weights.set(1.0, letter3, sub2from3);
  layer1weights.set(-1.0, letter2, sub2from3);
  // const sub3from2 = relu(1.0 * letter2 + -1.0 * letter3)
  layer1weights.set(1.0, letter2, sub3from2);
  layer1weights.set(-1.0, letter3, sub3from2);
  // const number1layer1 = relu(1.0 * number1)
  layer1weights.set(1.0, number1, number1layer1);
  // const number2layer1 = relu(1.0 * number2)
  layer1weights.set(1.0, number2, number2layer1);

  const /*mut*/ layer2weights = tf.buffer([neuronsPerLayer, neuronsPerLayer])
  const /*mut*/ layer2bias = tf.buffer([neuronsPerLayer]);
  const contribution1 = 0;
  const contribution2 = 1;
  // const contribution1 = relu(1.0 * number1layer1 + -1000.0 * sub1from3 + -1000.0 * sub3from1)
  layer2weights.set(1.0, number1layer1, contribution1);
  layer2weights.set(-1000.0, sub1from3, contribution1);
  layer2weights.set(-1000.0, sub3from1, contribution1);
  // const contribution2 = relu(1.0 * number2layer1 + -1000.0 * sub2from3 + -1000.0 * sub3from2)
  layer2weights.set(1.0, number2layer1, contribution2);
  layer2weights.set(-1000.0, sub2from3, contribution2);
  layer2weights.set(-1000.0, sub3from2, contribution2);

  const /*mut*/ layer3weights = tf.buffer([neuronsPerLayer, neuronsPerLayer])
  const /*mut*/ layer3bias = tf.buffer([neuronsPerLayer]);
  
  // Dynamically create neurons for each value in vocab
  // For each number 1..vocabSize, we need:
  //   - subNfromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - N)
  //   - subOutFromN = relu(-1.0 * contribution1 + -1.0 * contribution2 + N)
  const /*mut*/ subNfromOut: number[] = [];
  const /*mut*/ subOutFromN: number[] = [];
  
  for (let i = 0; i < currentVocabSize; i++) {
    const neuronValue = i + 1; // Values are 1-indexed
    
    // subNfromOut neurons (indices 0, 1, 2, ...)
    const subFromIndex = i;
    subNfromOut.push(subFromIndex);
    layer3weights.set(1.0, contribution1, subFromIndex);
    layer3weights.set(1.0, contribution2, subFromIndex);
    layer3bias.set(-neuronValue, subFromIndex);
    
    // subOutFromN neurons (indices vocabSize, vocabSize+1, vocabSize+2, ...)
    const subToIndex = currentVocabSize + i;
    subOutFromN.push(subToIndex);
    layer3weights.set(-1.0, contribution1, subToIndex);
    layer3weights.set(-1.0, contribution2, subToIndex);
    layer3bias.set(neuronValue, subToIndex);
  }

  const /*mut*/ layer4weights = tf.buffer([neuronsPerLayer, neuronsPerLayer])
  const /*mut*/ layer4bias = tf.buffer([neuronsPerLayer]);
  
  // Dynamically create probability neurons for each value in vocab
  // For each number 1..vocabSize:
  //   probabilityN = relu(-1.0 * subNFromOut + -1.0 * subOutFromN + 1.0)
  const /*mut*/ probabilityIndices: number[] = [];
  
  for (let i = 0; i < currentVocabSize; i++) {
    const probabilityIndex = i;
    probabilityIndices.push(probabilityIndex);
    
    // relu(-1.0 * subNFromOut + -1.0 * subOutFromN + 1.0)
    layer4weights.set(-1.0, subNfromOut[i], probabilityIndex);
    layer4weights.set(-1.0, subOutFromN[i], probabilityIndex);
    layer4bias.set(1.0, probabilityIndex);
  }

  // Layers 5 and beyond (if any) implement identity function on their first vocabSize
  // neurons (the probability outputs).
  const /*mut*/ extraLayerWeights: any[] = [];
  for (let layerIdx = 4; layerIdx < numLayers; layerIdx++) {
    const prevLayerSize = neuronsPerLayer;
    const currLayerSize = neuronsPerLayer;

    const weights = tf.buffer([prevLayerSize, currLayerSize]);
    const bias = tf.buffer([currLayerSize]);

    // Set identity connections for the first vocabSize neurons
    for (let i = 0; i < currentVocabSize; i++) {
      weights.set(1.0, probabilityIndices[i], probabilityIndices[i]);
    }

    extraLayerWeights.push(weights.toTensor(), bias.toTensor());
  }

  // At this point we have
  //   A=1 B=2 A=___
  //             P(A=1) = 1
  //             P(A=2) = 0
  //             P(A=3) = 0
  //   A=1 B=2 C=___
  //             P(B=1) = 0
  //             P(B=2) = 0
  //             P(B=3) = 0
  // which looks great but softmax will mess this up so we need to push P(A=1)
  // way up and P(A="A=") way down.

  // Output layer connects to the last hidden layer
  const outputSize = getOutputSize(currentVocabSize);
  const /*mut*/ outputWeights = tf.buffer([neuronsPerLayer, outputSize])
  const /*mut*/ outputBias = tf.buffer([outputSize]);
  
  // Set weights for each probability neuron to corresponding output
  for (let i = 0; i < currentVocabSize; i++) {
    outputWeights.set(1000.0, probabilityIndices[i], i);
  }
  
  // Set bias values
  for (let i = 0; i < outputSize; i++) {
    if (i < currentVocabSize) {
      outputBias.set(-100, i);
    } else {
      outputBias.set(-Infinity, i);
    }
  }

  const perfectWeights: Tensor[] = [
    layer1weights.toTensor(), layer1bias.toTensor(),
    layer2weights.toTensor(), layer2bias.toTensor(),
    layer3weights.toTensor(), layer3bias.toTensor(),
    layer4weights.toTensor(), layer4bias.toTensor(),
    ...extraLayerWeights,
    outputWeights.toTensor(), outputBias.toTensor()
  ];
  window.messageLoop({ type: "SetModelWeights", weights: perfectWeights } as SetModelWeightsMsg);

  await drawViz();
  perfectWeights.forEach(tensor => tensor.dispose());
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, newNumLayers, newNeuronsPerLayer, _inputFormat, vocabSize) => {
  numLayers = newNumLayers;
  neuronsPerLayer = newNeuronsPerLayer;
  currentVocabSize = vocabSize;
  updatePerfectWeightsButton();
};

export { 
  init,
  canUsePerfectWeights, 
  updatePerfectWeightsButton, 
  setPerfectWeights, 
  reinitializeModel 
};
