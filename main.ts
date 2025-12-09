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
    xsArray: number[];
    ysArray: number[];
}

// Define the model based on a shared configuration
const HIDDEN_LAYER_SIZES = [2, 2];

const EPOCHS_PER_BATCH = 10;

function createModel(): Sequential {
    const model = tf.sequential();

    // Add the first hidden layer with inputShape
    model.add(tf.layers.dense({
        units: HIDDEN_LAYER_SIZES[0],
        inputShape: [1],
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
        units: 1
    }));

    // Compile the model with mean squared error loss
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.sgd(0.001) // Stochastic Gradient Descent with a small learning rate
    });

    return model;
}

// Generate training data for y = 2x - 1
function generateData(): TrainingData {
    const xs: number[] = [];
    const ys: number[] = [];
    
    // Create 100 training examples
    for (let i = 0; i < 100; i++) {
        const x = Math.random() * 10 - 5; // Random values between -5 and 5
        const y = 2 * x - 1; // True relationship
        xs.push(x);
        ys.push(y);
    }
    
    // Convert to tensors
    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = tf.tensor2d(ys, [ys.length, 1]);
    
    return { xs: xsTensor, ys: ysTensor, xsArray: xs, ysArray: ys };
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
    await updatePredictionCurve();
    drawLossCurve();
    
    // Request the next frame
    requestAnimationFrame(trainingStep);
}

// --- Visualization Updates ---
async function updatePredictionCurve(): Promise<void> {
    const xRange = tf.linspace(-5, 5, 100);
    const yPredsTensor = model.predict(xRange.reshape([100, 1])) as Tensor;
    
    const xRangeArray = await xRange.array();
    const yPredsArray = (await yPredsTensor.array() as number[][]).flat();
    
    visualize(data.xsArray, data.ysArray, xRangeArray, yPredsArray);
    
    // Clean up tensors
    xRange.dispose();
    yPredsTensor.dispose();
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

    // Draw axes
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, canvas.height - 30);
    ctx.lineTo(canvas.width - 30, canvas.height - 30);
    ctx.moveTo(30, 30);
    ctx.lineTo(30, canvas.height - 30);
    ctx.stroke();

    // Draw loss curve
    ctx.strokeStyle = 'purple';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(lossHistory[0].epoch), toCanvasY(lossHistory[0].loss));
    for (let i = 1; i < lossHistory.length; i++) {
        ctx.lineTo(toCanvasX(lossHistory[i].epoch), toCanvasY(lossHistory[i].loss));
    }
    ctx.stroke();

    // Add labels
    ctx.font = '12px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.fillText(`Epoch: ${maxEpoch}`, canvas.width / 2, canvas.height - 10);
    ctx.save();
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`Loss: ${maxLoss.toFixed(4)}`, -canvas.height / 2, 15);
    ctx.restore();
}

// Visualize the data and learned model
function visualize(xs: number[], ys: number[], xPred: number[], yPred: number[]): void {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Find data range
    const xMin = -5;
    const xMax = 5;
    const yMin = -15;
    const yMax = 15;
    
    // Helper function to convert data coordinates to canvas coordinates
    function toCanvasX(x: number): number {
        return ((x - xMin) / (xMax - xMin)) * (canvas.width - 60) + 30;
    }
    
    function toCanvasY(y: number): number {
        return canvas.height - 30 - ((y - yMin) / (yMax - yMin)) * (canvas.height - 60);
    }
    
    // Draw axes
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, canvas.height - 30);
    ctx.lineTo(canvas.width - 30, canvas.height - 30);
    ctx.moveTo(30, 30);
    ctx.lineTo(30, canvas.height - 30);
    ctx.stroke();
    
    // Draw data points
    ctx.fillStyle = 'blue';
    for (let i = 0; i < xs.length; i++) {
        ctx.beginPath();
        ctx.arc(toCanvasX(xs[i]), toCanvasY(ys[i]), 3, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Draw learned function
    if (xPred.length > 0 && yPred.length > 0) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(xPred[0]), toCanvasY(yPred[0]));
        for (let i = 1; i < xPred.length; i++) {
            ctx.lineTo(toCanvasX(xPred[i]), toCanvasY(yPred[i]));
        }
        ctx.stroke();
    }
    
    // Draw true line (green, dashed)
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(toCanvasX(xMin), toCanvasY(2 * xMin - 1));
    ctx.lineTo(toCanvasX(xMax), toCanvasY(2 * xMax - 1));
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Add legend
    ctx.font = '14px Arial';
    ctx.fillStyle = 'blue';
    ctx.fillText('● Training data', canvas.width - 150, 20);
    ctx.fillStyle = 'red';
    ctx.fillText('— Learned model', canvas.width - 150, 40);
    ctx.fillStyle = 'green';
    ctx.fillText('- - True model (y=2x-1)', canvas.width - 200, 60);
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
    const statusElement = document.getElementById('status')!;
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;

    const requestedBackend = backendSelector.value;

    // GPU Fallback Logic
    if (requestedBackend === 'webgpu') {
        const fallbacks = ['webgpu', 'webgl', 'wasm', 'cpu'];
        for (const backend of fallbacks) {
            try {
                await tf.setBackend(backend);
                console.log(`TensorFlow.js backend set to: ${tf.getBackend()}`);
                statusElement.innerHTML = `Backend set to <strong>${tf.getBackend()}</strong>. Click the button to start training...`;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return; // Exit after successful backend setting
            } catch (error) {
                console.warn(`Failed to initialize ${backend} backend:`, error);
                if (backend === 'webgl' || backend === 'wasm') {
                    showError(`Failed to initialize ${backend}, falling back...`);
                }
            }
        }
        // If all fallbacks fail
        statusElement.innerHTML = `Error: Could not initialize GPU or any fallback backend.`;
        return;
    }

    // Default logic for other backends
    try {
        await tf.setBackend(requestedBackend);
        console.log(`TensorFlow.js backend set to: ${tf.getBackend()}`);
        statusElement.innerHTML = `Backend set to <strong>${tf.getBackend()}</strong>. Click the button to start training...`;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    } catch (error) {
        console.error(`Failed to set backend to ${requestedBackend}:`, error);
        statusElement.innerHTML = `Error: Could not initialize <strong>${requestedBackend}</strong> backend. Your browser may not support it.`;
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
    await setBackend();
    initializeNewModel();
    drawNetworkArchitecture();
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

    // Clear canvases and update UI
    const statusElement = document.getElementById('status')!;
    const lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;

    statusElement.innerHTML = 'Ready to train!';

    const lossCtx = lossCanvas.getContext('2d')!;
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

    // Visualize the initial (untrained) state
    updatePredictionCurve();
}

// Visualize the network architecture
function drawNetworkArchitecture(): void {
    const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerGap = 40;
    const nodeWidth = 20;
    const nodeHeight = 20;
    const layers = [1, ...HIDDEN_LAYER_SIZES, 1]; // Input, Hidden Layers, Output

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
