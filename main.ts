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
    
    // Add a single dense layer with 1 unit (output)
    // This creates a linear model: y = mx + b
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
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
    
    // Get the learned parameters
    const weights = model.getWeights();
    const slope = weights[0].dataSync()[0];
    const intercept = weights[1].dataSync()[0];
    
    // Make predictions
    const testX = tf.tensor2d([0, 1, 2, 3, 4], [5, 1]);
    const predictions = model.predict(testX) as Tensor;
    const predArray = await predictions.array() as number[][];
    
    // Display results
    statusElement.innerHTML = 'Training complete!';
    resultsElement.style.display = 'block';
    paramsElement.innerHTML = 
        `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`;
    
    let predText = '<strong>Sample predictions:</strong><br>';
    for (let i = 0; i < 5; i++) {
        predText += `x = ${i}, predicted y = ${predArray[i][0].toFixed(4)}, actual y = ${2 * i - 1}<br>`;
    }
    predictionsElement.innerHTML = predText;
    
    // Visualize
    visualize(data.xsArray, data.ysArray, slope, intercept);
    
    // Clean up tensors
    data.xs.dispose();
    data.ys.dispose();
    testX.dispose();
    predictions.dispose();
}

// Visualize the data and learned model
function visualize(xs: number[], ys: number[], slope: number, intercept: number): void {
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
    
    // Draw learned line
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(xMin), toCanvasY(slope * xMin + intercept));
    ctx.lineTo(toCanvasX(xMax), toCanvasY(slope * xMax + intercept));
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

// Make trainModel available globally
(window as any).trainModel = trainModel;
