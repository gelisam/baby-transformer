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
let data: TrainingData;
let isTraining = false;
let currentEpoch = 0;
const lossHistory: { epoch: number, loss: number }[] = [];

interface TrainingData {
  xs: Tensor2D;
  ys: Tensor2D;
}

const INPUT_SIZE = 5;
const HIDDEN_LAYER_SIZES = [4, 2, 3];
const OUTPUT_SIZE = 7;  // Vocabulary: A, B, C, =, 1, 2, 3

const EPOCHS_PER_BATCH = 10;

function createModel(): Sequential {
  const model = tf.sequential();

  // Add the first hidden layer with inputShape
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

  // Compile the model with mean squared error loss
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });

  return model;
}

// Generate training data for the classification task
// Vocabulary: 0=A, 1=B, 2=C, 3==, 4=1, 5=2, 6=3
function generateData(): TrainingData {
  const A = 0, B = 1, C = 2, EQ = 3, ONE = 4, TWO = 5, THREE = 6;
  
  const allInputs: number[][] = [];
  const allOutputs: number[] = [];

  // Generate all valid sequences of the form: letter=number letter=number ...
  // where the same letter always maps to the same number within a sequence
  
  // We'll generate examples with different mappings
  // Each mapping is a function from {A,B,C} to {1,2,3}
  const letters = [A, B, C];
  const numbers = [ONE, TWO, THREE];
  
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
  for (const numPerm of numberPerms) {
    const mapping = new Map<number, number>();
    mapping.set(A, numPerm[0]);
    mapping.set(B, numPerm[1]);
    mapping.set(C, numPerm[2]);
    
    // Generate sequences using this mapping
    // We need sequences of 5 tokens (input) + 1 token (output)
    // Pattern: L1 = N1 L2 = N2 ... where we predict the next token
    
    // Generate all possible sequences of up to 3 pairs
    for (let numPairs = 1; numPairs <= 3; numPairs++) {
      // Generate all sequences of `numPairs` letter choices
      function generateSequences(length: number, choices: number[]): number[][] {
        if (length === 0) return [[]];
        const result: number[][] = [];
        const subSeqs = generateSequences(length - 1, choices);
        for (const choice of choices) {
          for (const subSeq of subSeqs) {
            result.push([choice, ...subSeq]);
          }
        }
        return result;
      }
      
      const letterSequences = generateSequences(numPairs, letters);
      
      for (const letterSeq of letterSequences) {
        // Build the full token sequence
        const fullSeq: number[] = [];
        for (const letter of letterSeq) {
          fullSeq.push(letter);
          fullSeq.push(EQ);
          fullSeq.push(mapping.get(letter)!);
        }
        
        // Create training examples by sliding a window
        // Input: 5 tokens, Output: 6th token
        for (let i = 0; i + INPUT_SIZE < fullSeq.length; i++) {
          const input = fullSeq.slice(i, i + INPUT_SIZE);
          const output = fullSeq[i + INPUT_SIZE];
          allInputs.push(input);
          allOutputs.push(output);
        }
      }
    }
  }

  // Convert to tensors
  const xsData = allInputs.flat();
  const numExamples = allInputs.length;
  const xs = tf.tensor2d(xsData, [numExamples, INPUT_SIZE], 'int32');
  const xsOneHot = tf.oneHot(xs, OUTPUT_SIZE);
  
  const ys = tf.oneHot(tf.tensor1d(allOutputs, 'int32'), OUTPUT_SIZE);

  xs.dispose();

  return {
    xs: xsOneHot as Tensor2D,
    ys: ys as Tensor2D
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
  const history = await model.fit(data.xs, data.ys, {
    epochs: EPOCHS_PER_BATCH,
    verbose: 0
  });

  currentEpoch += EPOCHS_PER_BATCH;

  // Get the loss from the last epoch in the batch
  const loss = history.history.loss[history.history.loss.length - 1] as number;
  statusElement.innerHTML = `Training... Epoch ${currentEpoch} - Loss: ${loss.toFixed(4)}`;

  lossHistory.push({ epoch: currentEpoch, loss });
  await drawOutput();
  drawLossCurve();

  // Request the next frame
  requestAnimationFrame(trainingStep);
}

// --- Visualization Updates ---
async function drawOutput(): Promise<void> {
  const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;

  // Token vocabulary: 0=A, 1=B, 2=C, 3==, 4=1, 5=2, 6=3
  const tokenNames = ["A", "B", "C", "=", "1", "2", "3"];
  
  // Generate 6 random valid input sequences
  const A = 0, B = 1, C = 2, EQ = 3, ONE = 4, TWO = 5, THREE = 6;
  const letters = [A, B, C];
  const numbers = [ONE, TWO, THREE];
  
  const testInputs: number[][] = [];
  
  // Helper to shuffle array
  function shuffle<T>(array: T[]): T[] {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  
  // Generate 6 random examples with different mappings
  for (let i = 0; i < 6; i++) {
    const mapping = new Map<number, number>();
    const shuffledNumbers = shuffle(numbers);
    mapping.set(A, shuffledNumbers[0]);
    mapping.set(B, shuffledNumbers[1]);
    mapping.set(C, shuffledNumbers[2]);
    
    // Pick a random sequence of letters
    const seqLength = 2 + Math.floor(Math.random() * 2); // 2 or 3 pairs
    const letterSeq: number[] = [];
    for (let j = 0; j < seqLength; j++) {
      letterSeq.push(letters[Math.floor(Math.random() * letters.length)]);
    }
    
    // Build the full sequence
    const fullSeq: number[] = [];
    for (const letter of letterSeq) {
      fullSeq.push(letter);
      fullSeq.push(EQ);
      fullSeq.push(mapping.get(letter)!);
    }
    
    // Take the first 5 tokens as input
    testInputs.push(fullSeq.slice(0, INPUT_SIZE));
  }

  // Convert to tensor and get predictions
  const inputsTensor = tf.tensor2d(testInputs, [testInputs.length, INPUT_SIZE], 'int32');
  const inputsOneHot = tf.oneHot(inputsTensor, OUTPUT_SIZE);
  const predictions = model.predict(inputsOneHot) as Tensor;
  const predictionsArray = await predictions.array() as number[][];

  // Clear canvas only after predictions are ready to avoid flickering
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const numRows = 2;
  const numCols = 3;
  const sectionSpacing = 10;
  const barSpacing = 3;

  const availableWidth = canvas.width - (sectionSpacing * (numCols + 1));
  const sectionWidth = availableWidth / numCols;
  const availableHeight = canvas.height - (sectionSpacing * (numRows + 1));
  const sectionHeight = availableHeight / numRows;

  // Set style for section borders
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;

  for (let i = 0; i < testInputs.length; i++) {
    const col = i % numCols;
    const row = Math.floor(i / numCols);

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    // Draw thin black border around section
    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    // Display the input sequence
    const inputTokens = testInputs[i].map(t => tokenNames[t]).join(' ');
    ctx.font = '12px monospace';
    ctx.fillStyle = 'black';
    ctx.fillText(inputTokens, sectionX + 5, sectionY + 15);

    const probabilities = predictionsArray[i];
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
      ctx.fillText(tokenNames[j], barX, sectionY + sectionHeight - 5);
    }
  }

  inputsTensor.dispose();
  inputsOneHot.dispose();
  predictions.dispose();
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
      data.xs.dispose();
      data.ys.dispose();
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

  // Reset training state
  currentEpoch = 0;
  lossHistory.length = 0;

  const statusElement = document.getElementById('status')!;
  statusElement.innerHTML = 'Ready to train!';

  // Visualize the initial (untrained) state
  drawOutput();
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

// Make trainModel available globally
(window as any).trainModel = trainModel;
