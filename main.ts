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

const INPUT_SIZE = 8;
const HIDDEN_LAYER_SIZES = [5, 2, 4];
const OUTPUT_SIZE = 7;

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
    // Generate some synthetic data for the new architecture
    const numSamples = 100;
    const allInputs = tf.randomUniform([numSamples, INPUT_SIZE]);
    const allOutputs = tf.randomUniform([numSamples, OUTPUT_SIZE]);

    // Ensure the data is in a one-hot format for classification
    const xs = tf.oneHot(tf.argMax(allInputs, 1), INPUT_SIZE);
    const ys = tf.oneHot(tf.argMax(allOutputs, 1), OUTPUT_SIZE);

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

    const allInputs = tf.oneHot(tf.tensor1d(Array.from(Array(INPUT_SIZE).keys()), 'int32'), INPUT_SIZE);
    const predictions = model.predict(allInputs) as Tensor;
    const predictionsArray = await predictions.array() as number[][];

    const labels = Array.from(Array(INPUT_SIZE).keys()).map(i => `Input ${i + 1}`);
    const numRows = 2;
    const numCols = Math.ceil(INPUT_SIZE / numRows);
    const sectionWidth = canvas.width / numCols;
    const sectionHeight = canvas.height / numRows;

    for (let i = 0; i < INPUT_SIZE; i++) {
        const col = i % numCols;
        const row = Math.floor(i / numCols);

        const sectionX = col * sectionWidth;
        const sectionY = row * sectionHeight;

        ctx.font = '12px Arial';
        ctx.fillStyle = 'black';
        ctx.fillText(labels[i], sectionX + sectionWidth / 2 - 20, sectionY + 15);

        const probabilities = predictionsArray[i];
        const barWidth = sectionWidth / (probabilities.length * 1.5);

        for (let j = 0; j < probabilities.length; j++) {
            const barHeight = probabilities[j] * (sectionHeight - 40);
            const barX = sectionX + (j * barWidth * 1.5) + (barWidth / 2);
            const barY = sectionY + sectionHeight - barHeight - 20;

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

    // Draw connections
    ctx.strokeStyle = 'lightgray';
    ctx.lineWidth = 1;

    for (let i = 0; i < layers.length - 1; i++) {
        const currentLayerNeurons = layers[i];
        const nextLayerNeurons = layers[i+1];
        const currentGeom = layerGeometries[i];
        const nextGeom = layerGeometries[i+1];

        const currentNeuronsX = Array.from({length: currentLayerNeurons}, (_, j) => currentGeom.x + (j / (currentLayerNeurons - 1)) * currentGeom.width);
        const nextNeuronsX = Array.from({length: nextLayerNeurons}, (_, j) => nextGeom.x + (j / (nextLayerNeurons - 1)) * nextGeom.width);

        const startY = currentGeom.y + currentGeom.height;
        const endY = nextGeom.y;

        const smallerLayerSize = Math.min(currentLayerNeurons, nextLayerNeurons);
        const largerLayerSize = Math.max(currentLayerNeurons, nextLayerNeurons);

        const isCurrentLayerSmaller = currentLayerNeurons < nextLayerNeurons;

        const smallerNeuronsX = isCurrentLayerSmaller ? currentNeuronsX : nextNeuronsX;
        const largerNeuronsX = isCurrentLayerSmaller ? nextNeuronsX : currentNeuronsX;

        const middleSectionSize = Math.floor(smallerLayerSize / 2) * 2; // Largest even number <= smallerLayerSize
        const leftoverSize = largerLayerSize - middleSectionSize;
        const sideSize = Math.floor(leftoverSize / 2);

        // --- Draw middle connections ---
        // Case 1: Layers have the same parity (even-even or odd-odd)
        if ((currentLayerNeurons % 2 === 0 && nextLayerNeurons % 2 === 0) || (currentLayerNeurons % 2 !== 0 && nextLayerNeurons % 2 !== 0)) {
            // Draw X-shaped connections for pairs of neurons.
            for (let j = 0; j < middleSectionSize / 2; j++) {
                const idx1 = j * 2;
                const idx2 = j * 2 + 1;

                const startX1 = isCurrentLayerSmaller ? smallerNeuronsX[idx1] : largerNeuronsX[sideSize + idx1];
                const startX2 = isCurrentLayerSmaller ? smallerNeuronsX[idx2] : largerNeuronsX[sideSize + idx2];
                const endX1 = isCurrentLayerSmaller ? largerNeuronsX[sideSize + idx1] : smallerNeuronsX[idx1];
                const endX2 = isCurrentLayerSmaller ? largerNeuronsX[sideSize + idx2] : smallerNeuronsX[idx2];

                // Draw the two lines that form the "X"
                ctx.beginPath();
                ctx.moveTo(startX1, startY);
                ctx.lineTo(endX2, endY);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(startX2, startY);
                ctx.lineTo(endX1, endY);
                ctx.stroke();
            }
        } else {
            // Case 2: Layers have different parity (even-odd or odd-even)
            // Draw a fan-out connection pattern from the middle neurons.
            for (let j = 0; j < middleSectionSize; j++) {
                const startX = isCurrentLayerSmaller ? smallerNeuronsX[j] : largerNeuronsX[sideSize + j];
                const endX = isCurrentLayerSmaller ? largerNeuronsX[sideSize + j] : smallerNeuronsX[j];

                // Each neuron connects straight to its counterpart.
                ctx.beginPath();
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
                ctx.stroke();

                // Each neuron also connects to the next neuron in the opposite layer to create the fan effect.
                if (j < middleSectionSize - 1) {
                    const nextEndX = isCurrentLayerSmaller ? largerNeuronsX[sideSize + j + 1] : smallerNeuronsX[j + 1];
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(nextEndX, endY);
                    ctx.stroke();
                }
            }
        }

        // --- Draw side connections ---
        const firstSmallerX = smallerNeuronsX[0];
        const lastSmallerX = smallerNeuronsX[smallerLayerSize - 1];

        // Left side
        for(let j=0; j<sideSize; ++j){
            ctx.beginPath();
            ctx.moveTo(isCurrentLayerSmaller ? firstSmallerX : largerNeuronsX[j], startY);
            ctx.lineTo(isCurrentLayerSmaller ? largerNeuronsX[j] : firstSmallerX, endY);
            ctx.stroke();
        }

        // Right side
        for(let j=0; j<sideSize; ++j){
            ctx.beginPath();
            ctx.moveTo(isCurrentLayerSmaller ? lastSmallerX : largerNeuronsX[largerLayerSize - 1 - j], startY);
            ctx.lineTo(isCurrentLayerSmaller ? largerNeuronsX[largerLayerSize - 1 - j] : lastSmallerX, endY);
            ctx.stroke();
        }

        // Connect the middle leftover neuron if it exists
        if(leftoverSize % 2 !== 0){
             const middleX = largerNeuronsX[sideSize + middleSectionSize];
             const smallerLayerIndex = middleSectionSize > 0 ? middleSectionSize -1 : 0;

             ctx.beginPath();
             ctx.moveTo(isCurrentLayerSmaller ? smallerNeuronsX[smallerLayerIndex] : middleX, startY);
             ctx.lineTo(isCurrentLayerSmaller ? middleX : smallerNeuronsX[smallerLayerIndex], endY);
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
