import {
  NUMBERS,
  LETTERS,
  tokenNumberToIndex,
  tokenStringToTokenNumber,
  tokenNumberToTokenString
} from "./tokens.js";
import {
  INPUT_SIZE,
  OUTPUT_SIZE,
  EPOCHS_PER_BATCH,
} from "./constants.js";
import {
  EMBEDDING_DIM,
  EMBEDDED_INPUT_SIZE,
  embedInput
} from "./embeddings.js";
import { createModel } from "./model.js";
import { tf, Tensor2D, Sequential } from "./tf.js";
import { pickRandomInputs, updateVizDataFromTextboxes, drawViz, drawLossCurve, drawNetworkArchitecture, VIZ_EXAMPLES_COUNT, VIZ_COLUMNS, VIZ_ROWS } from "./viz.js";
import { TrainingData, AppState } from "./types.js";



//////////////////
// Global State //
//////////////////

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



// Generate training data for the classification task
function generateData(): TrainingData {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];
  function addExample(sequence: number[]) {
    //console.log(sequence.map(tokenNumberToTokenString).join(''));
    const input = sequence.slice(0, INPUT_SIZE);
    const output = sequence[INPUT_SIZE];
    inputArray.push(input);
    outputArray.push(output);
  }

  function removeElementAt(arr: number[], index: number): number[] {
    const newArr = arr.slice(); // ["a", "b", "c"]
    newArr.splice(index, 1); // returns ["b"] and mutates 'newArr' to ["a", "c"]
    return newArr;
  }

  function insert(mapping: Map<number, number>, key: number, value: number): Map<number, number> {
    const newMapping = new Map(mapping);
    newMapping.set(key, value);
    return newMapping;
  }

  const allLetters = LETTERS.map(tokenStringToTokenNumber);
  const allNumbers = NUMBERS.map(tokenStringToTokenNumber);
  function generate(
      n: number, // number of examples to generate before the final pair
      sequence: number[],
      mapping: Map<number, number>,
      availableLetters: number[],
      availableNumbers: number[]
  ) {
    if (n === 0) {
      for (const letter of allLetters) {
        if (mapping.has(letter)) {
          addExample([...sequence, letter, mapping.get(letter)!]);
        } else {
          for (const number of availableNumbers) {
            addExample([...sequence, letter, number]);
          }
        }
      }
    } else {
      for (let i = 0; i < availableLetters.length; i++) {
        const letter = availableLetters[i];
        const newAvailableLetters = removeElementAt(availableLetters, i);
        for (let j = 0; j < availableNumbers.length; j++) {
          const number = availableNumbers[j];
          const newAvailableNumbers = removeElementAt(availableNumbers, j);
          const newMapping = insert(mapping, letter, number);
          generate(n - 1, [...sequence, letter, number], newMapping, newAvailableLetters, newAvailableNumbers);
        }
      }
    }
  }

  generate(2, [], new Map(), allLetters, allNumbers);

  // Convert to tensors with embeddings
  const numExamples = inputArray.length;
  const embeddedInputArray = inputArray.map(embedInput);
  
  const inputTensor = tf.tensor2d(embeddedInputArray, [numExamples, EMBEDDED_INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  return {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };
}

// --- Training Control ---
function updateLayerConfiguration(numLayers: number, neuronsPerLayer: number): void {
  // Stop training and reinitialize model
  if (appState.isTraining) {
    toggleTrainingMode(); // Toggles isTraining to false
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

  initializeNewModel();
  updatePerfectWeightsButton();
}

function canUsePerfectWeights(numLayers: number, neuronsPerLayer: number): { canUse: boolean, reason: string } {
  // The setPerfectWeights function needs to be updated for the new embedding/unembedding architecture
  return {
    canUse: false,
    reason: 'Perfect weights feature not yet implemented for embedding/unembedding architecture.'
  };
}

function updatePerfectWeightsButton(): void {
  const button = document.getElementById('perfect-weights-button') as HTMLButtonElement;
  const tooltipText = document.getElementById('perfect-weights-tooltip-text') as HTMLSpanElement;
  const result = canUsePerfectWeights(appState.num_layers, appState.neurons_per_layer);

  button.disabled = !result.canUse;

  if (!result.canUse) {
    tooltipText.textContent = result.reason;
  } else {
    tooltipText.textContent = '';
  }
}

async function toggleTrainingMode() {
  appState.isTraining = !appState.isTraining;
  const trainButton = document.getElementById('train-button')!;

  if (appState.isTraining) {
    trainButton.innerText = 'Pause';
    requestAnimationFrame(trainingStep);
  } else {
    trainButton.innerText = 'Train Model';
  }
}

async function trainingStep() {
  if (!appState.isTraining) {
    // Training has been paused
    return;
  }

  const statusElement = document.getElementById('status')!;

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
  await drawViz(appState, appState.vizData);
  drawLossCurve(appState);

  // Request the next frame
  requestAnimationFrame(trainingStep);
}

async function setPerfectWeights(): Promise<void> {
  if (appState.isTraining) {
    toggleTrainingMode(); // Toggles isTraining to false
  }

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

  const /*mut*/ layer1weights = tf.buffer([INPUT_SIZE, appState.neurons_per_layer])
  const /*mut*/ layer1bias = tf.buffer([appState.neurons_per_layer]);
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

  const /*mut*/ layer2weights = tf.buffer([appState.neurons_per_layer, appState.neurons_per_layer])
  const /*mut*/ layer2bias = tf.buffer([appState.neurons_per_layer]);
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

  const /*mut*/ layer3weights = tf.buffer([appState.neurons_per_layer, appState.neurons_per_layer])
  const /*mut*/ layer3bias = tf.buffer([appState.neurons_per_layer]);
  const sub1fromOut = 0;
  const sub2fromOut = 1;
  const sub3fromOut = 2;
  const subOutFrom1 = 3;
  const subOutFrom2 = 4;
  const subOutFrom3 = 5;
  // const sub1fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 1.0)
  layer3weights.set(1.0, contribution1, sub1fromOut);
  layer3weights.set(1.0, contribution2, sub1fromOut);
  layer3bias.set(-1.0, sub1fromOut);
  // const sub2fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 2.0)
  layer3weights.set(1.0, contribution1, sub2fromOut);
  layer3weights.set(1.0, contribution2, sub2fromOut);
  layer3bias.set(-2.0, sub2fromOut);
  // const sub3fromOut = relu(1.0 * contribution1 + 1.0 * contribution2 - 3.0)
  layer3weights.set(1.0, contribution1, sub3fromOut);
  layer3weights.set(1.0, contribution2, sub3fromOut);
  layer3bias.set(-3.0, sub3fromOut);
  // const subOutFrom1 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 1.0)
  layer3weights.set(-1.0, contribution1, subOutFrom1);
  layer3weights.set(-1.0, contribution2, subOutFrom1);
  layer3bias.set(1.0, subOutFrom1);
  // const subOutFrom2 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 2.0)
  layer3weights.set(-1.0, contribution1, subOutFrom2);
  layer3weights.set(-1.0, contribution2, subOutFrom2);
  layer3bias.set(2.0, subOutFrom2);
  // const subOutFrom3 = relu(-1.0 * contribution1 + -1.0 * contribution2 + 3.0)
  layer3weights.set(-1.0, contribution1, subOutFrom3);
  layer3weights.set(-1.0, contribution2, subOutFrom3);
  layer3bias.set(3.0, subOutFrom3);

  const /*mut*/ layer4weights = tf.buffer([appState.neurons_per_layer, appState.neurons_per_layer])
  const /*mut*/ layer4bias = tf.buffer([appState.neurons_per_layer]);
  const probability1 = 0;
  const probability2 = 1;
  const probability3 = 2;
  // relu(-1.0 * sub1FromOut + -1.0 * subOutFrom1 + 1.0),
  layer4weights.set(-1.0, sub1fromOut, probability1);
  layer4weights.set(-1.0, subOutFrom1, probability1);
  layer4bias.set(1.0, probability1);
  // relu(-1.0 * sub2FromOut + -1.0 * subOutFrom2 + 1.0),
  layer4weights.set(-1.0, sub2fromOut, probability2);
  layer4weights.set(-1.0, subOutFrom2, probability2);
  layer4bias.set(1.0, probability2);
  // relu(-1.0 * sub3FromOut + -1.0 * subOutFrom3 + 1.0),
  layer4weights.set(-1.0, sub3fromOut, probability3);
  layer4weights.set(-1.0, subOutFrom3, probability3);
  layer4bias.set(1.0, probability3);

  // Layers 5 and beyond (if any) implement identity function on their first 3
  // inputs.
  const extraLayerWeights: any[] = [];
  for (let layerIdx = 4; layerIdx < appState.num_layers; layerIdx++) {
    const prevLayerSize = appState.neurons_per_layer;
    const currLayerSize = appState.neurons_per_layer;

    const weights = tf.buffer([prevLayerSize, currLayerSize]);
    const bias = tf.buffer([currLayerSize]);

    // Set identity connections for the first 3 neurons
    weights.set(1.0, probability1, probability1);
    weights.set(1.0, probability2, probability2);
    weights.set(1.0, probability3, probability3);

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
  const /*mut*/ outputWeights = tf.buffer([appState.neurons_per_layer, OUTPUT_SIZE])
  const /*mut*/ outputBias = tf.buffer([OUTPUT_SIZE]);
  outputWeights.set(1000.0, probability1, probability1);
  outputWeights.set(1000.0, probability2, probability2);
  outputWeights.set(1000.0, probability3, probability3);
  outputBias.set(-100, 0);
  outputBias.set(-100, 1);
  outputBias.set(-100, 2);
  outputBias.set(-Infinity, 3);
  outputBias.set(-Infinity, 4);
  outputBias.set(-Infinity, 5);

  const perfectWeights = [
    layer1weights.toTensor(), layer1bias.toTensor(),
    layer2weights.toTensor(), layer2bias.toTensor(),
    layer3weights.toTensor(), layer3bias.toTensor(),
    layer4weights.toTensor(), layer4bias.toTensor(),
    ...extraLayerWeights,
    outputWeights.toTensor(), outputBias.toTensor()
  ];
  appState.model.setWeights(perfectWeights);

  await drawViz(appState, appState.vizData);
  perfectWeights.forEach(tensor => tensor.dispose());
}

// --- Backend Selection Logic ---

// Function to display toaster-style error messages
function showError(message: string): void {
  const toaster = document.getElementById('toaster')!;
  toaster.textContent = message;
  toaster.style.display = 'block';

  // Hide the toaster after 3 seconds
  setTimeout(() => {
    toaster.style.display = 'none';
  }, 3000);
}

// Function to set the backend and update UI
async function setBackend() {
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  const requestedBackend = backendSelector.value;

  try {
    await tf.setBackend(requestedBackend);
    console.log(`TensorFlow.js backend set to: ${tf.getBackend()}`);
  } catch (error) {
    console.error(`Failed to set backend to ${requestedBackend}:`, error);
  }
}

// Set up backend selection when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
  const trainButton = document.getElementById('train-button') as HTMLButtonElement;
  trainButton.addEventListener('click', toggleTrainingMode);
  const perfectWeightsButton = document.getElementById('perfect-weights-button') as HTMLButtonElement;
  perfectWeightsButton.addEventListener('click', setPerfectWeights);

  // Add event listener for changes
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  backendSelector.addEventListener('change', async () => {
    // Stop training and clean up old tensors before changing backend
    if (appState.isTraining) {
      toggleTrainingMode(); // Toggles isTraining to false
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
    updateLayerConfiguration(appState.num_layers, appState.neurons_per_layer);
  });

  neuronsPerLayerSlider.addEventListener('input', () => {
    appState.neurons_per_layer = parseInt(neuronsPerLayerSlider.value, 10);
    neuronsPerLayerValue.textContent = appState.neurons_per_layer.toString();
    updateLayerConfiguration(appState.num_layers, appState.neurons_per_layer);
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
  updatePerfectWeightsButton();
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
