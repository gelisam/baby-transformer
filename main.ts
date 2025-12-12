/// <reference types="@tensorflow/tfjs" />

// TensorFlow.js is loaded via CDN in index.html
declare const tf: typeof import('@tensorflow/tfjs');

// Import types only (no runtime import)
type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;
type Logs = import('@tensorflow/tfjs').Logs;
type Tensor = import('@tensorflow/tfjs').Tensor;

// --- Global State ---
let model: Sequential;
let isTraining = false;
let currentEpoch = 0;
let lossHistory: { epoch: number, loss: number }[] = [];

interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}
let data: TrainingData;

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;
let vizData: TrainingData;


const SHORT_NUMBERS = ["1", "2", "3"];
const SHORT_LETTERS = ["A", "B", "C"];
const SHORT_TOKENS = [...SHORT_NUMBERS, ...SHORT_LETTERS];
const NUMBERS = SHORT_NUMBERS.map(n => n + " ");
const LETTERS = SHORT_LETTERS.map(l => l + "=");
const TOKENS = [...NUMBERS, ...LETTERS];

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const HIDDEN_LAYER_SIZES = [4, 2, 3];
const OUTPUT_SIZE = TOKENS.length;

const EPOCHS_PER_BATCH = 1;

const TOKEN_STRING_TO_INDEX: { [key: string]: number } = {};
for (let i = 0; i < TOKENS.length; i++) {
  TOKEN_STRING_TO_INDEX[TOKENS[i]] = i;
}
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

function createModel(): Sequential {
  const model = tf.sequential();

  // Add the first hidden layer with inputShape
  // Input is one-hot encoded, so each token becomes OUTPUT_SIZE dimensions
  model.add(tf.layers.dense({
    units: HIDDEN_LAYER_SIZES[0],
    inputShape: [INPUT_SIZE],
    activation: 'relu'
  }));

  // Add the remaining hidden layers
  for (let i = 1; i < HIDDEN_LAYER_SIZES.length; i++) {
    model.add(tf.layers.dense({
      units: HIDDEN_LAYER_SIZES[i],
      activation: 'relu'
    }));
  }

  // Add the linear output layer
  model.add(tf.layers.dense({
    units: OUTPUT_SIZE,
    activation: 'softmax'
  }));

  // Compile the model with categorical cross-entropy loss
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });

  return model;
}

// Generate training data for the classification task
// Vocabulary: 0=A, 1=B, 2=C, 3='=', 4=1, 5=2, 6=3
function generateData(): TrainingData {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  // Generate all valid sequences of the form: letter=number letter=number ...
  // where the same letter always maps to the same number within a sequence

  // We'll generate examples with different mappings
  // Each mapping is a function from {A,B,C} to {1,2,3}
  const letters: number[] = LETTERS.map(tokenStringToTokenNumber);
  const numbers: number[] = NUMBERS.map(tokenStringToTokenNumber);

  // Generate all possible mappings (permutations)
  function generatePermutations(arr: number[]): number[][] {
    if (arr.length <= 1) return [arr];
    const result: number[][] = [];
    for (let i = 0; i < arr.length; i++) {
      const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
      const perms = generatePermutations(rest);
      for (const perm of perms) {
        result.push([arr[i], ...perm]);
      }
    }
    return result;
  }

  const numberPerms = generatePermutations(numbers);

  // For each permutation (mapping), generate various sequences
  for (const perm of numberPerms) {
    const mapping = new Map<number, number>();
    for (let i = 0; i < letters.length; i++) {
      mapping.set(letters[i], perm[i]);
    }

    // Generate sequences using this mapping
    // We need sequences of 5 tokens (input) + 1 token (output)
    // input: <letter>=<number>  <letter>=<number> <letter>=
    // output:                                              <number>

    // Generate n <letter>=<number> pairs from the mapping
    function generateSequences(length: number): number[][] {
      if (length === 0) return [[]];
      const result: number[][] = [];
      const subSeqs = generateSequences(length - 1);
      for (const letter of letters) {
        const n = mapping.get(letter)!;
        for (const subSeq of subSeqs) {
          result.push([letter, n, ...subSeq]);
        }
      }
      return result;
    }

    const numPairs = 3;
    const sequences = generateSequences(numPairs);

    for (const sequence of sequences) {
      const input = sequence.slice(0, INPUT_SIZE);
      const output = sequence[INPUT_SIZE];
      inputArray.push(input);
      outputArray.push(output);
    }
  }

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
async function trainModel() {
  isTraining = !isTraining;
  const trainButton = document.getElementById('train-button')!;

  if (isTraining) {
    trainButton.innerText = 'Pause';
    requestAnimationFrame(trainingStep);
  } else {
    trainButton.innerText = 'Train Model';
  }
}

async function trainingStep() {
  if (!isTraining) {
    // Training has been paused
    return;
  }

  const statusElement = document.getElementById('status')!;

  // Train for one epoch
  const history = await model.fit(data.inputTensor, data.outputTensor, {
    epochs: EPOCHS_PER_BATCH,
    verbose: 0
  });

  currentEpoch += EPOCHS_PER_BATCH;

  // Get the loss from the last epoch in the batch
  const loss = history.history.loss[history.history.loss.length - 1] as number;
  statusElement.innerHTML = `Training... Epoch ${currentEpoch} - Loss: ${loss.toFixed(4)}`;

  lossHistory.push({ epoch: currentEpoch, loss });
  await drawViz(vizData);
  drawLossCurve();

  // Request the next frame
  requestAnimationFrame(trainingStep);
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
  return {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };
}

async function drawViz(vizData: TrainingData): Promise<void> {
  const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;

  const inputArray = vizData.inputArray;
  const outputArray = vizData.outputArray;
  const inputTensor = vizData.inputTensor;

  // Get predictions
  const predictionTensor = model.predict(inputTensor) as Tensor2D;
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
  if (lossHistory.length < 2) {
    return;
  }

  const canvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Find data range
  const minLoss = Math.min(...lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...lossHistory.map(d => d.loss));
  const minEpoch = lossHistory[0].epoch;
  const maxEpoch = lossHistory[lossHistory.length - 1].epoch;

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
  ctx.moveTo(toCanvasX(lossHistory[0].epoch), toCanvasY(lossHistory[0].loss));
  for (let i = 1; i < lossHistory.length; i++) {
    ctx.lineTo(toCanvasX(lossHistory[i].epoch), toCanvasY(lossHistory[i].loss));
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
    if (isTraining) {
      trainModel(); // Toggles isTraining to false
    }
    if (data) {
      data.inputTensor.dispose();
      data.outputTensor.dispose();
    }
    if (vizData) {
      vizData.inputTensor.dispose();
      vizData.outputTensor.dispose();
    }

    await setBackend();
    initializeNewModel(); // Initialize a new model for the new backend
  });

  // Initial setup
  drawNetworkArchitecture();
  await setBackend();
  initializeNewModel();
});

function initializeNewModel(): void {
  // Create a new model
  if (model) {
    model.dispose();
  }
  model = createModel();

  // Generate new data
  // No need to clean up old data tensors here, it's handled on backend change
  data = generateData();

  // Generate visualization inputs (only once, not on every frame)
  vizData = pickRandomInputs(data);

  // Reset training state
  currentEpoch = 0;
  lossHistory.length = 0;

  const statusElement = document.getElementById('status')!;
  statusElement.innerHTML = 'Ready to train!';

  // Visualize the initial (untrained) state
  drawViz(vizData);
}

// Visualize the network architecture
function drawNetworkArchitecture(): void {
  const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const layers = [INPUT_SIZE, ...HIDDEN_LAYER_SIZES, OUTPUT_SIZE];
  const layerHeight = 30; // Height of the rectangle for each layer
  const maxLayerWidth = canvas.width * 0.4; // Max width for a layer
  const layerGapY = 50; // Vertical gap between layers
  const startY = 30;
  const canvasWidth = canvas.width;

  const maxNeurons = Math.max(...layers);

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
      const y = currentLayer.y + currentLayer.height;
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

    // Draw layer rectangle
    ctx.fillStyle = 'darkblue';
    ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

    // Draw layer label
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    let label = '';
    if (i === 0) {
      label = 'Input';
    } else if (i === layers.length - 1) {
      label = 'Linear Output';
    } else {
      label = `Hidden Layer ${i}`;
    }

    const labelText = `${label} (${numNeurons} neurons)`;
    ctx.fillText(labelText, canvasWidth - 20, geom.y + geom.height / 2);
  }
}
