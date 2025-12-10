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
const HIDDEN_LAYER_SIZES = [10];
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
    const inputs = [1, 2, 3, 4, 5, 6];
    const outputs = [4, 5, 6, 1, 2, 3];

    const xs = tf.oneHot(tf.tensor1d(inputs.map(i => i - 1), 'int32'), OUTPUT_SIZE);
    const ys = tf.oneHot(tf.tensor1d(outputs.map(o => o - 1), 'int32'), OUTPUT_SIZE);

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
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const allInputs = tf.oneHot(tf.tensor1d([0, 1, 2, 3, 4, 5], 'int32'), OUTPUT_SIZE);
    const predictions = model.predict(allInputs) as Tensor;
    const predictionsArray = await predictions.array() as number[][];

    const labels = ["1", "2", "3", "4", "5", "6"];
    const numRows = 2;
    const numCols = 3;
    const sectionWidth = canvas.width / numCols;
    const sectionHeight = canvas.height / numRows;

    for (let i = 0; i < labels.length; i++) {
        const col = i % numCols;
        // The first 3 items (0, 1, 2) go to the bottom row (row 1)
        // The next 3 items (3, 4, 5) go to the top row (row 0)
        const row = i < 3 ? 1 : 0;

        const sectionX = col * sectionWidth;
        const sectionY = row * sectionHeight;

        ctx.font = '16px Arial';
        ctx.fillStyle = 'black';
        ctx.fillText(labels[i], sectionX + sectionWidth / 2 - 10, sectionY + 20);

        const probabilities = predictionsArray[i];
        const barWidth = sectionWidth / (probabilities.length * 1.5);

        for (let j = 0; j < probabilities.length; j++) {
            const barHeight = probabilities[j] * (sectionHeight - 60);
            const barX = sectionX + (j * barWidth * 1.5) + (barWidth / 2);
            const barY = sectionY + sectionHeight - barHeight - 30;

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

    const layerGap = 40;
    const nodeWidth = 20;
    const nodeHeight = 20;
    const layers = [INPUT_SIZE, ...HIDDEN_LAYER_SIZES, OUTPUT_SIZE]; // Input, Hidden Layers, Output

    // Function to draw a node (filled rectangle)
    function drawNode(x: number, y: number): void {
        ctx.fillStyle = 'black';
        ctx.fillRect(x - nodeWidth / 2, y - nodeHeight / 2, nodeWidth, nodeHeight);
    }

    // Function to draw a connection
    function drawConnection(x1: number, y1: number, x2: number, y2: number): void {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = 'gray';
        ctx.stroke();
    }

    const startY = 30;
    const canvasWidth = canvas.width;

    // Store neuron positions
    const neuronPositions: { x: number, y: number }[][] = [];

    // First, calculate all neuron positions
    for (let i = 0; i < layers.length; i++) {
        const layerY = startY + i * layerGap;
        const numNeurons = layers[i];
        const layerPositions: { x: number, y: number }[] = [];

        for (let j = 0; j < numNeurons; j++) {
            const neuronX = (canvasWidth / 2) - ((numNeurons - 1) * 60 / 2) + j * 60;
            layerPositions.push({ x: neuronX, y: layerY });
        }
        neuronPositions.push(layerPositions);
    }

    // Then, draw connections
    for (let i = 0; i < neuronPositions.length - 1; i++) {
        for (const pos1 of neuronPositions[i]) {
            for (const pos2 of neuronPositions[i + 1]) {
                drawConnection(pos1.x, pos1.y, pos2.x, pos2.y);
            }
        }
    }

    // Finally, draw neurons and labels on top
    for (let i = 0; i < neuronPositions.length; i++) {
        const layerY = neuronPositions[i][0].y;
        for (const pos of neuronPositions[i]) {
            drawNode(pos.x, pos.y);
        }

        // Draw layer labels on the right
        ctx.fillStyle = 'black';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        let label = '';
        if (i === 0) label = 'Input';
        else if (i === layers.length - 1) label = 'Linear Output';
        else label = `Hidden Layer ${i}`;
        ctx.fillText(label, canvasWidth - 120, layerY);
    }
}

// Make trainModel available globally
(window as any).trainModel = trainModel;
