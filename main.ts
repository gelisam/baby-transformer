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

const INPUT_SIZE = 6;
const HIDDEN_LAYER_SIZES = [4, 4, 2, 3];
const OUTPUT_SIZE = 6;

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
function generateData(): TrainingData {
  const inputs1 = [1, 2, 3];
  const outputs1 = [4, 5, 6];
  const inputs2 = [4, 5, 6];
  const outputs2 = [1, 2, 3];

  const allInputs: number[] = [];
  const allOutputs: number[] = [];

  // First set of mappings
  for (const input of inputs1) {
    for (const output of outputs1) {
      allInputs.push(input);
      allOutputs.push(output);
    }
  }

  // Second set of mappings
  for (const input of inputs2) {
    for (const output of outputs2) {
      allInputs.push(input);
      allOutputs.push(output);
    }
  }

  const xs = tf.oneHot(tf.tensor1d(allInputs.map(i => i - 1), 'int32'), OUTPUT_SIZE);
  const ys = tf.oneHot(tf.tensor1d(allOutputs.map(o => o - 1), 'int32'), OUTPUT_SIZE);

  return {
    xs: xs as Tensor2D,
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

  const allInputs = tf.oneHot(tf.tensor1d([0, 1, 2, 3, 4, 5], 'int32'), OUTPUT_SIZE);
  const predictions = model.predict(allInputs) as Tensor;
  const predictionsArray = await predictions.array() as number[][];
  
  // Clear canvas only after predictions are ready to avoid flickering
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const labels = ["1 ", "2 ", "3 ", "A=", "B=", "C="];
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

  for (let i = 0; i < labels.length; i++) {
    const col = i % numCols;
    // The first 3 items (0, 1, 2) go to the bottom row (row 1)
    // The next 3 items (3, 4, 5) go to the top row (row 0)
    const row = i < 3 ? 1 : 0;

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    // Draw thin black border around section
    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    ctx.font = '16px Arial';
    ctx.fillStyle = 'black';
    ctx.fillText(labels[i], sectionX + sectionWidth / 2 - 10, sectionY + 20);

    const probabilities = predictionsArray[i];
    const numBars = probabilities.length;
    const barWidth = (sectionWidth - barSpacing * (numBars + 1)) / numBars;

    for (let j = 0; j < probabilities.length; j++) {
      const barHeight = probabilities[j] * (sectionHeight - 60);
      const barX = sectionX + barSpacing + j * (barWidth + barSpacing);
      const barY = sectionY + sectionHeight - barHeight - barSpacing;

      ctx.fillStyle = 'blue';
      ctx.fillRect(barX, barY, barWidth, barHeight);
    }
  }

  allInputs.dispose();
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
    
    // Check if both have same parity (both even or both odd)
    const bothEvenOrOdd = (currentNeurons % 2) === (nextNeurons % 2);
    
    if (bothEvenOrOdd) {
      // Use X-shaped pairs between consecutive neurons from smaller to larger
      // Leftover neurons are at the edges of the larger layer
      const leftoverCount = largerCount - smallerCount;
      const leftoverLeft = Math.floor(leftoverCount / 2);
      const leftoverRight = leftoverCount - leftoverLeft;
      
      // Connect leftmost smaller neuron to leftover left neurons
      for (let l = 0; l < leftoverLeft; l++) {
        ctx.beginPath();
        ctx.moveTo(smallerPositions[0].x, smallerPositions[0].y);
        ctx.lineTo(largerPositions[l].x, largerPositions[l].y);
        ctx.stroke();
      }
      
      // Draw X patterns for pairs in the middle
      const pairsFromSmaller = Math.floor(smallerCount / 2);
      for (let p = 0; p < pairsFromSmaller; p++) {
        const small1 = p * 2;
        const small2 = p * 2 + 1;
        const large1 = leftoverLeft + p * 2;
        const large2 = leftoverLeft + p * 2 + 1;
        
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
      
      // Connect rightmost smaller neuron to leftover right neurons
      for (let r = 0; r < leftoverRight; r++) {
        ctx.beginPath();
        ctx.moveTo(smallerPositions[smallerCount - 1].x, smallerPositions[smallerCount - 1].y);
        ctx.lineTo(largerPositions[largerCount - leftoverRight + r].x, largerPositions[largerCount - leftoverRight + r].y);
        ctx.stroke();
      }
    } else {
      // One odd, one even: zig-zag in the middle
      const smallerIsOdd = smallerCount % 2 === 1;
      
      if (smallerIsOdd) {
        // Smaller is odd, larger is even
        const middleIdx = Math.floor(smallerCount / 2);
        
        // Process each neuron in smaller layer
        for (let s = 0; s < smallerCount; s++) {
          if (s < middleIdx) {
            // Left side: X pattern
            const large1 = s * 2;
            const large2 = s * 2 + 1;
            
            ctx.beginPath();
            ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
            ctx.lineTo(largerPositions[large1].x, largerPositions[large1].y);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
            ctx.lineTo(largerPositions[large2].x, largerPositions[large2].y);
            ctx.stroke();
          } else if (s === middleIdx) {
            // Middle: zig-zag
            const large1 = s * 2;
            const large2 = s * 2 + 1;
            
            if (large1 < largerCount) {
              ctx.beginPath();
              ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
              ctx.lineTo(largerPositions[large1].x, largerPositions[large1].y);
              ctx.stroke();
            }
            
            if (large2 < largerCount) {
              ctx.beginPath();
              ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
              ctx.lineTo(largerPositions[large2].x, largerPositions[large2].y);
              ctx.stroke();
            }
          } else {
            // Right side: X pattern
            const offset = middleIdx * 2 + 2;
            const localIdx = s - middleIdx - 1;
            const large1 = offset + localIdx * 2;
            const large2 = offset + localIdx * 2 + 1;
            
            if (large1 < largerCount) {
              ctx.beginPath();
              ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
              ctx.lineTo(largerPositions[large1].x, largerPositions[large1].y);
              ctx.stroke();
            }
            
            if (large2 < largerCount) {
              ctx.beginPath();
              ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
              ctx.lineTo(largerPositions[large2].x, largerPositions[large2].y);
              ctx.stroke();
            }
          }
        }
      } else {
        // Smaller is even, larger is odd
        const middleIdx = largerCount / 2;
        
        // Each smaller neuron connects to corresponding larger neurons
        for (let s = 0; s < smallerCount; s++) {
          const ratio = largerCount / smallerCount;
          const start = Math.floor(s * ratio);
          const end = Math.ceil((s + 1) * ratio);
          
          for (let l = start; l < end && l < largerCount; l++) {
            ctx.beginPath();
            ctx.moveTo(smallerPositions[s].x, smallerPositions[s].y);
            ctx.lineTo(largerPositions[l].x, largerPositions[l].y);
            ctx.stroke();
          }
        }
      }
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
