/// <reference types="@tensorflow/tfjs" />

// TensorFlow.js is loaded via CDN in index.html
declare const tf: typeof import('@tensorflow/tfjs');

// Import types only (no runtime import)
type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;
type Logs = import('@tensorflow/tfjs').Logs;
type Tensor = import('@tensorflow/tfjs').Tensor;

interface TrainingData {
    xs: Tensor2D;
    ys: Tensor2D;
    xsArray: number[];
    ysArray: number[];
}

// Define the model
function createModel(): Sequential {
    const model = tf.sequential();

    // Two hidden layers with ReLU activation
    model.add(tf.layers.dense({
        units: 2,
        inputShape: [1],
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 2,
        activation: 'relu'
    }));

    // Linear output layer
    model.add(tf.layers.dense({
        units: 1
    }));

    // Compile the model with mean squared error loss
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.sgd(0.1) // Stochastic Gradient Descent with learning rate 0.1
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

// Train the model
async function trainModel(): Promise<void> {
    const statusElement = document.getElementById('status')!;
    const resultsElement = document.getElementById('results')!;
    const paramsElement = document.getElementById('params')!;
    const predictionsElement = document.getElementById('predictions')!;
    
    statusElement.innerHTML = 'Training model...';
    resultsElement.style.display = 'none';
    
    // Create model
    const model = createModel();
    
    // Generate data
    const data = generateData();
    
    // Train the model
    await model.fit(data.xs, data.ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch: number, logs?: Logs) => {
                if (epoch % 10 === 0 && logs) {
                    statusElement.innerHTML = 
                        `Training... Epoch ${epoch}/100 - Loss: ${logs.loss.toFixed(4)}`;
                }
            }
        }
    });
    
    // Get the learned parameters - This is no longer a simple line
    // We will visualize the learned function by predicting over a range
    
    // Make predictions for visualization
    const xRange = tf.linspace(-5, 5, 100);
    const yPreds = model.predict(xRange.reshape([100, 1])) as Tensor;
    const xRangeArray = await xRange.array();
    const yPredsArray = await yPreds.array();

    // Make predictions for the table
    const testX = tf.tensor2d([0, 1, 2, 3, 4], [5, 1]);
    const predictions = model.predict(testX) as Tensor;
    const predArray = await predictions.array() as number[][];
    
    // Display results
    statusElement.innerHTML = 'Training complete!';
    resultsElement.style.display = 'block';
    paramsElement.innerHTML = `The model is a 2x2 ReLU network with a linear output.`;
    
    let predText = '<strong>Sample predictions:</strong><br>';
    for (let i = 0; i < 5; i++) {
        predText += `x = ${i}, predicted y = ${predArray[i][0].toFixed(4)}, actual y = ${2 * i - 1}<br>`;
    }
    predictionsElement.innerHTML = predText;
    
    // Visualize
    visualize(data.xsArray, data.ysArray, xRangeArray, (yPredsArray as number[][]).flat());
    drawNetworkArchitecture();
    
    // Clean up tensors
    data.xs.dispose();
    data.ys.dispose();
    testX.dispose();
    predictions.dispose();
    xRange.dispose();
    yPreds.dispose();
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
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(xPred[0]), toCanvasY(yPred[0]));
    for (let i = 1; i < xPred.length; i++) {
        ctx.lineTo(toCanvasX(xPred[i]), toCanvasY(yPred[i]));
    }
    ctx.stroke();
    
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

// Function to set the backend and update UI
async function setBackend() {
    const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
    const statusElement = document.getElementById('status')!;
    const resultsElement = document.getElementById('results')!;
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;

    const backend = backendSelector.value;

    try {
        await tf.setBackend(backend);
        console.log(`TensorFlow.js backend set to: ${tf.getBackend()}`);

        // Reset UI elements
        statusElement.innerHTML = `Backend set to <strong>${tf.getBackend()}</strong>. Click the button to start training...`;
        resultsElement.style.display = 'none';
        ctx.clearRect(0, 0, canvas.width, canvas.height);

    } catch (error) {
        console.error(`Failed to set backend to ${backend}:`, error);
        // Handle cases where a backend is not available (e.g., WebGPU)
        statusElement.innerHTML = `Error: Could not initialize <strong>${backend}</strong> backend. Your browser may not support it.`;
    }
}

// Set up backend selection when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;

    // Set the initial backend
    await setBackend();

    // Add event listener for changes
    backendSelector.addEventListener('change', setBackend);
});

// Visualize the network architecture
function drawNetworkArchitecture(): void {
    const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerGap = 150;
    const neuronRadius = 15;
    const layers = [1, 2, 2, 1]; // Input, Hidden 1, Hidden 2, Output

    // Function to draw a neuron
    function drawNeuron(x: number, y: number, text: string): void {
        ctx.beginPath();
        ctx.arc(x, y, neuronRadius, 0, 2 * Math.PI);
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.stroke();
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x, y);
    }

    // Function to draw a connection
    function drawConnection(x1: number, y1: number, x2: number, y2: number): void {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = 'gray';
        ctx.stroke();
    }

    const startX = 50;
    const canvasHeight = canvas.height;

    // Store neuron positions
    const neuronPositions: { x: number, y: number }[][] = [];

    // Draw neurons and labels
    for (let i = 0; i < layers.length; i++) {
        const layerX = startX + i * layerGap;
        const numNeurons = layers[i];
        const layerPositions: { x: number, y: number }[] = [];

        for (let j = 0; j < numNeurons; j++) {
            const neuronY = (canvasHeight - (numNeurons - 1) * 60) / 2 + j * 60;
            layerPositions.push({ x: layerX, y: neuronY });

            let neuronText = '';
            if (i > 0 && i < layers.length - 1) {
                neuronText = 'ReLU';
            }
            drawNeuron(layerX, neuronY, neuronText);
        }
        neuronPositions.push(layerPositions);

        // Draw layer labels
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        let label = '';
        if (i === 0) label = 'Input';
        else if (i === layers.length - 1) label = 'Linear Output';
        else label = `Hidden Layer ${i}`;
        ctx.fillText(label, layerX, canvasHeight - 20);
    }

    // Draw connections
    for (let i = 0; i < neuronPositions.length - 1; i++) {
        for (const pos1 of neuronPositions[i]) {
            for (const pos2 of neuronPositions[i + 1]) {
                drawConnection(pos1.x, pos1.y, pos2.x, pos2.y);
            }
        }
    }
}

// Make trainModel available globally
(window as any).trainModel = trainModel;
