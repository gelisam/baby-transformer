/// <reference types="@tensorflow/tfjs" />

// TensorFlow.js is loaded via CDN in index.html
declare const tf: typeof import('@tensorflow/tfjs');

// Import types only (no runtime import)
type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;
type Logs = import('@tensorflow/tfjs').Logs;
type Tensor = import('@tensorflow/tfjs').Tensor;

// --- Configuration Constants ---
const SHORT_NUMBERS = ["1", "2", "3"];
const SHORT_LETTERS = ["A", "B", "C"];
const SHORT_TOKENS = [...SHORT_NUMBERS, ...SHORT_LETTERS];
const NUMBERS = SHORT_NUMBERS.map(n => n + " ");
const LETTERS = SHORT_LETTERS.map(l => l + "=");
const TOKENS = [...NUMBERS, ...LETTERS];

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const OUTPUT_SIZE = TOKENS.length;

const EPOCHS_PER_BATCH = 1;

const TOKEN_STRING_TO_INDEX: { [key: string]: number } = {};
for (let i = 0; i < TOKENS.length; i++) {
  TOKEN_STRING_TO_INDEX[TOKENS[i]] = i;
}

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

// --- Global State ---
interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

const appState = {
  model: undefined as unknown as Sequential,
  isTraining: false,
  currentEpoch: 0,
  lossHistory: [] as { epoch: number, loss: number }[],
  data: undefined as unknown as TrainingData,
  vizData: undefined as unknown as TrainingData,
  num_layers: 4,
  neurons_per_layer: 6
};
function indexToTokenNumber(index: number): number {
  return index + 1;
}
function indexToShortTokenString(index: number): string {
  return SHORT_TOKENS[index];
}
function indexToTokenString(index: number): string {
  return TOKENS[index];
}
function tokenNumberToIndex(tokenNum: number): number {
  return tokenNum - 1;
}
function tokenStringToIndex(token: string): number {
  return TOKEN_STRING_TO_INDEX[token];
}
function tokenStringToTokenNumber(token: string): number {
  return TOKEN_STRING_TO_INDEX[token] + 1;
}
function tokenNumberToTokenString(tokenNum: number): string {
  return TOKENS[tokenNum - 1];
}

function createModel(numLayers: number, neuronsPerLayer: number): Sequential {
  const model = tf.sequential();

  if (numLayers === 0) {
    model.add(tf.layers.dense({
      units: OUTPUT_SIZE,
      inputShape: [INPUT_SIZE],
      activation: 'relu'
    }));
  } else {
    // Add the first hidden layer with inputShape
    model.add(tf.layers.dense({
      units: neuronsPerLayer,
      inputShape: [INPUT_SIZE],
      activation: 'relu'
    }));

    // Add the remaining hidden layers
    for (let i = 1; i < numLayers; i++) {
      model.add(tf.layers.dense({
        units: neuronsPerLayer,
        activation: 'relu'
      }));
    }

    // Add the linear output layer
    model.add(tf.layers.dense({
      units: OUTPUT_SIZE,
      activation: 'softmax'
    }));
  }

  // Compile the model with categorical cross-entropy loss
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });

  return model;
}

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

  // Convert to tensors
  const numExamples = inputArray.length;
  const inputTensor = tf.tensor2d(inputArray, [numExamples, INPUT_SIZE]);
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
  // The setPerfectWeights function requires at least 4 hidden layers with at least 6 neurons per layer
  // Extra layers beyond the first 4 implement identity functions on their first 3 inputs
  // Extra neurons beyond the minimum are set to implement identity functions
  const minRequiredLayers = 4;
  const minRequiredNeurons = 6;

  if (numLayers < minRequiredLayers) {
    return {
      canUse: false,
      reason: `Requires at least ${minRequiredLayers} hidden layers, but currently configured with ${numLayers}.`
    };
  }

  if (neuronsPerLayer < minRequiredNeurons) {
    return {
      canUse: false,
      reason: `Requires at least ${minRequiredNeurons} neurons per layer, but currently configured with ${neuronsPerLayer}.`
    };
  }

  return { canUse: true, reason: '' };
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
  await drawViz(appState.vizData);
  drawLossCurve();

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

  await drawViz(appState.vizData);
  perfectWeights.forEach(tensor => tensor.dispose());
}

// --- Visualization Updates ---

// Pick random test inputs for visualization (called once at initialization)
function pickRandomInputs(data: TrainingData): TrainingData {
  let inputArray: number[][] = [];
  let outputArray: number[] = [];
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const randomIndex = Math.floor(Math.random() * data.inputArray.length);
    inputArray.push(data.inputArray[randomIndex]);
    outputArray.push(data.outputArray[randomIndex]);
  }
  const inputTensor = tf.tensor2d(inputArray, [VIZ_EXAMPLES_COUNT, INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  // Populate the textboxes with these random inputs
  updateTextboxesFromInputs(inputArray, outputArray);

  return {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };
}

// Update textboxes with the current inputs
function updateTextboxesFromInputs(inputArray: number[][], outputArray: number[]): void {
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = document.getElementById(`input-${i}`) as HTMLInputElement;
    if (inputElement) {
      const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
      inputElement.value = inputTokenStrings;
    }
  }
}

// Parse input string into token numbers
function parseInputString(inputStr: string): number[] | null {
  const tokens: number[] = [];
  let i = 0;

  while (i < inputStr.length) {
    let matched = false;

    // Try to match each token
    for (const token of TOKENS) {
      if (inputStr.substring(i, i + token.length) === token) {
        tokens.push(tokenStringToTokenNumber(token));
        i += token.length;
        matched = true;
        break;
      }
    }

    if (!matched) {
      return null; // Invalid input
    }
  }

  return tokens.length === INPUT_SIZE ? tokens : null;
}

// Update visualization data from textboxes
function updateVizDataFromTextboxes(): void {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = document.getElementById(`input-${i}`) as HTMLInputElement;
    if (inputElement) {
      const parsed = parseInputString(inputElement.value);
      if (parsed) {
        inputArray.push(parsed);
        // Find the corresponding output from the original data
        const matchingIndex = appState.data.inputArray.findIndex(arr =>
          arr.every((val, idx) => val === parsed[idx])
        );
        if (matchingIndex >= 0) {
          outputArray.push(appState.data.outputArray[matchingIndex]);
        } else {
          // If not found in training data, use a default output
          outputArray.push(tokenStringToTokenNumber(NUMBERS[0]));
        }
      } else {
        // If invalid, keep the previous value or use a default
        if (appState.vizData && appState.vizData.inputArray[i]) {
          inputArray.push(appState.vizData.inputArray[i]);
          outputArray.push(appState.vizData.outputArray[i]);
        } else {
          // Fallback: use the first valid input from training data
          inputArray.push(appState.data.inputArray[0]);
          outputArray.push(appState.data.outputArray[0]);
        }
      }
    }
  }

  // Dispose old tensors
  if (appState.vizData) {
    appState.vizData.inputTensor.dispose();
    appState.vizData.outputTensor.dispose();
  }

  // Create new vizData
  const inputTensor = tf.tensor2d(inputArray, [VIZ_EXAMPLES_COUNT, INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  appState.vizData = {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };

  // Redraw visualization
  drawViz(appState.vizData);
}

async function drawViz(vizData: TrainingData): Promise<void> {
  const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;

  const inputArray = vizData.inputArray;
  const outputArray = vizData.outputArray;
  const inputTensor = vizData.inputTensor;

  // Get predictions
  const predictionTensor = appState.model.predict(inputTensor) as Tensor2D;
  const predictionArray = await predictionTensor.array() as number[][];

  // Clear canvas only after predictions are ready to avoid flickering
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const sectionSpacing = 10;
  const barSpacing = 3;

  const availableWidth = canvas.width - (sectionSpacing * (VIZ_COLUMNS + 1));
  const sectionWidth = availableWidth / VIZ_COLUMNS;
  const availableHeight = canvas.height - (sectionSpacing * (VIZ_ROWS + 1));
  const sectionHeight = availableHeight / VIZ_ROWS;

  // Set style for section borders
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;

  for (let i = 0; i < inputArray.length; i++) {
    const col = i % VIZ_COLUMNS;
    const row = Math.floor(i / VIZ_COLUMNS);

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    // Draw thin black border around section
    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    // Display the input sequence
    const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
    ctx.font = '12px monospace';
    ctx.fillStyle = 'black';
    ctx.fillText(inputTokenStrings, sectionX + 5, sectionY + 15);

    const probabilities = predictionArray[i];
    const numBars = probabilities.length;
    const barWidth = (sectionWidth - barSpacing * (numBars + 1)) / numBars;

    // Draw probability bars for each possible next token
    for (let j = 0; j < probabilities.length; j++) {
      const barHeight = probabilities[j] * (sectionHeight - 40);
      const barX = sectionX + barSpacing + j * (barWidth + barSpacing);
      const barY = sectionY + sectionHeight - barHeight - barSpacing - 15;

      ctx.fillStyle = 'blue';
      ctx.fillRect(barX, barY, barWidth, barHeight);

      // Draw token label below bar
      ctx.font = '10px monospace';
      ctx.fillStyle = 'black';
      ctx.fillText(indexToShortTokenString(j), barX, sectionY + sectionHeight - 5);
    }
  }

  predictionTensor.dispose();
}

function drawLossCurve(): void {
  if (appState.lossHistory.length < 2) {
    return;
  }

  const canvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Find data range
  const minLoss = Math.min(...appState.lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...appState.lossHistory.map(d => d.loss));
  const minEpoch = appState.lossHistory[0].epoch;
  const maxEpoch = appState.lossHistory[appState.lossHistory.length - 1].epoch;

  // Helper functions to convert data coordinates to canvas coordinates
  function toCanvasX(epoch: number): number {
    return ((epoch - minEpoch) / (maxEpoch - minEpoch)) * (canvas.width - 60) + 30;
  }

  function toCanvasY(loss: number): number {
    const range = maxLoss - minLoss;
    // Add a small epsilon to the range to avoid division by zero if all losses are the same
    const effectiveRange = range === 0 ? 1 : range;
    return canvas.height - 30 - ((loss - minLoss) / effectiveRange) * (canvas.height - 60);
  }

  // Draw loss curve
  ctx.strokeStyle = 'lightgrey';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toCanvasX(appState.lossHistory[0].epoch), toCanvasY(appState.lossHistory[0].loss));
  for (let i = 1; i < appState.lossHistory.length; i++) {
    ctx.lineTo(toCanvasX(appState.lossHistory[i].epoch), toCanvasY(appState.lossHistory[i].loss));
  }
  ctx.stroke();
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
        updateVizDataFromTextboxes();
      });
    }
  }

  // Initial setup
  drawNetworkArchitecture();
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
  drawViz(appState.vizData);

  // Redraw the architecture in case it changed
  drawNetworkArchitecture();
}

// Visualize the network architecture
function drawNetworkArchitecture(): void {
  const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const layers = [INPUT_SIZE, ...Array(appState.num_layers).fill(appState.neurons_per_layer), OUTPUT_SIZE];
  const layerHeight = 20; // Height of the rectangle for each layer
  const maxLayerWidth = canvas.width * 0.4; // Max width for a layer
  const layerGapY = 40; // Vertical gap between layers
  const startY = 30;
  const canvasWidth = canvas.width;
  const arrowHeadSize = 8; // Size of arrow heads

  const maxNeurons = Math.max(...layers);

  /**
   * Helper function to draw a thick downward arrow
   * @param ctx - The canvas rendering context
   * @param x - The horizontal center position of the arrow
   * @param startY - The Y coordinate where the arrow shaft starts
   * @param endY - The Y coordinate where the arrow head points to
   */
  function drawDownwardArrow(ctx: CanvasRenderingContext2D, x: number, startY: number, endY: number): void {
    ctx.lineWidth = 6;
    ctx.strokeStyle = 'darkblue';
    ctx.fillStyle = 'darkblue';

    // Draw arrow shaft
    ctx.beginPath();
    ctx.moveTo(x, startY);
    ctx.lineTo(x, endY - arrowHeadSize);
    ctx.stroke();

    // Draw arrow head
    ctx.beginPath();
    ctx.moveTo(x, endY);
    ctx.lineTo(x - arrowHeadSize, endY - arrowHeadSize);
    ctx.lineTo(x + arrowHeadSize, endY - arrowHeadSize);
    ctx.closePath();
    ctx.fill();
  }

  // Store layer positions and sizes
  const layerGeometries: { x: number, y: number, width: number, height: number }[] = [];

  // Calculate geometries for each layer
  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const layerWidth = (numNeurons / maxNeurons) * maxLayerWidth;
    const layerX = (canvasWidth / 2) - (layerWidth / 2);
    const layerY = startY + i * layerGapY;
    layerGeometries.push({ x: layerX, y: layerY, width: layerWidth, height: layerHeight });
  }

  // Draw connections (subset of edges between layers)
  ctx.strokeStyle = 'gray';
  ctx.lineWidth = 2;
  for (let i = 0; i < layerGeometries.length - 1; i++) {
    const currentLayer = layerGeometries[i];
    const nextLayer = layerGeometries[i + 1];
    const currentNeurons = layers[i];
    const nextNeurons = layers[i + 1];

    // Calculate positions of individual neurons in each layer
    const currentNeuronPositions: { x: number, y: number }[] = [];
    for (let n = 0; n < currentNeurons; n++) {
      const x = currentLayer.x + (currentLayer.width / currentNeurons) * (n + 0.5);
      const y = currentLayer.y + currentLayer.height + 1;
      currentNeuronPositions.push({ x, y });
    }

    const nextNeuronPositions: { x: number, y: number }[] = [];
    for (let n = 0; n < nextNeurons; n++) {
      const x = nextLayer.x + (nextLayer.width / nextNeurons) * (n + 0.5);
      const y = nextLayer.y;
      nextNeuronPositions.push({ x, y });
    }

    // Determine which layer is smaller
    const smallerCount = Math.min(currentNeurons, nextNeurons);
    const largerCount = Math.max(currentNeurons, nextNeurons);
    const isCurrentSmaller = currentNeurons <= nextNeurons;
    const smallerPositions = isCurrentSmaller ? currentNeuronPositions : nextNeuronPositions;
    const largerPositions = isCurrentSmaller ? nextNeuronPositions : currentNeuronPositions;

    // Leftover neurons are at the edges of the larger layer
    const leftoverCount = Math.ceil((largerCount - smallerCount) / 2);
    const leftoverLeft = leftoverCount;
    const leftoverRight = leftoverCount;

    // Connect leftmost smaller neuron to leftover left neurons
    for (let l = 0; l < leftoverLeft; l++) {
      ctx.beginPath();
      ctx.moveTo(smallerPositions[0].x, smallerPositions[0].y);
      ctx.lineTo(largerPositions[l].x, largerPositions[l].y);
      ctx.stroke();
    }

    // Check if both have same parity (both even or both odd)
    const bothEvenOrOdd = (currentNeurons % 2) === (nextNeurons % 2);

    if (bothEvenOrOdd) {
      // Draw X patterns for pairs in the middle
      const pairCount = smallerCount - 1;
      for (let p = 0; p < pairCount; p++) {
        const small1 = p;
        const small2 = p + 1;
        const large1 = leftoverLeft + p;
        const large2 = leftoverLeft + p + 1;

        // X pattern: small1 -> large2, small2 -> large1
        ctx.beginPath();
        ctx.moveTo(smallerPositions[small1].x, smallerPositions[small1].y);
        ctx.lineTo(largerPositions[large2].x, largerPositions[large2].y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(smallerPositions[small2].x, smallerPositions[small2].y);
        ctx.lineTo(largerPositions[large1].x, largerPositions[large1].y);
        ctx.stroke();
      }
    } else {
      // Draw a zig-zag in the middle
      const zigCount = smallerCount - 1;
      for (let z = 0; z < zigCount; z++) {
        const small1 = z;
        const small2 = z + 1;
        const large = leftoverLeft + z;

        // zig-zag pattern: small1 -> large -> small2
        ctx.beginPath();
        ctx.moveTo(smallerPositions[small1].x, smallerPositions[small1].y);
        ctx.lineTo(largerPositions[large].x, largerPositions[large].y);
        ctx.lineTo(smallerPositions[small2].x, smallerPositions[small2].y);
        ctx.stroke();
      }
    }

    // Connect rightmost smaller neuron to leftover right neurons
    for (let r = 0; r < leftoverRight; r++) {
      ctx.beginPath();
      ctx.moveTo(smallerPositions[smallerCount - 1].x, smallerPositions[smallerCount - 1].y);
      ctx.lineTo(largerPositions[largerCount - leftoverRight + r].x, largerPositions[largerCount - leftoverRight + r].y);
      ctx.stroke();
    }
  }

  // Draw layers and their labels
  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const geom = layerGeometries[i];
    const isHidden = i > 0 && i < layers.length - 1;

    if (i === 0) {
      // Input layer: draw only the bottom edge of an unfilled rectangle
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'darkblue';
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height);
      ctx.stroke();

      // Draw thick downward arrow in the middle of the rectangle
      const arrowX = geom.x + geom.width / 2;
      const arrowStartY = geom.y;
      const arrowEndY = geom.y + geom.height - 2;

      drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
    } else {
      // Draw main layer rectangle for non-input layers
      ctx.fillStyle = 'darkblue';
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

      // Color-code the bottom border for activation functions
      ctx.lineWidth = 4;
      if (isHidden) {
        // Hidden layers use ReLU
        ctx.strokeStyle = '#4682B4'; // SteelBlue for ReLU
        ctx.beginPath();
        ctx.moveTo(geom.x, geom.y + geom.height - 1);
        ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
        ctx.stroke();
      } else if (i === layers.length - 1) {
        // Output layer is followed by Softmax
        ctx.strokeStyle = 'rgba(255, 165, 0, 1)'; // Orange for Softmax
        ctx.beginPath();
        ctx.moveTo(geom.x, geom.y + geom.height - 1);
        ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
        ctx.stroke();
      }
    }

    // Draw layer label
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    let label = '';
    if (i === 0) {
      label = `${numNeurons}-wide input`;
    } else if (i === layers.length - 1) {
      label = `${numNeurons}-wide linear+softmax layer`;
    } else {
      label = `${numNeurons}-wide ReLU layer`;
    }

    ctx.fillText(label, canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }

  // Draw thick downward arrow after the output layer
  const geom = layerGeometries[layerGeometries.length - 1];
  const arrowX = geom.x + geom.width / 2;
  const arrowStartY = geom.y + layerHeight + 3;
  const arrowEndY = geom.y + layerHeight + 3 + geom.height - 2;

  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}
