import {
  INPUT_SIZE,
  OUTPUT_SIZE
} from "../constants.js";
import {
  EMBEDDED_INPUT_SIZE,
  EMBEDDING_DIM,
  embedInput,
  embedTokenNumber
} from "../embeddings.js";
import { tf, Tensor2D } from "../tf.js";
import {
  NUMBERS,
  TOKENS,
  indexToShortTokenString,
  tokenNumberToIndex,
  tokenNumberToTokenString,
  tokenStringToTokenNumber
} from "../tokens.js";
import { Schedule } from "../messageLoop.js";
import { OnEpochCompletedHandler } from "../messages/onEpochCompleted.js";
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetTrainingDataHandler } from "../messages/setTrainingData.js";
import { getModel, getLossHistory } from "./model.js";

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

// Module-local state for DOM elements (initialized on first use)
let outputCanvas: HTMLCanvasElement | null = null;
let lossCanvas: HTMLCanvasElement | null = null;
let networkCanvas: HTMLCanvasElement | null = null;
let inputElements: HTMLInputElement[] = [];
let statusElement: HTMLElement | null = null;
let domInitialized = false;

// Module-local state for visualization data
let vizInputArray: number[][] = [];
let vizOutputArray: number[] = [];
let vizInputTensor: Tensor2D | null = null;
let vizOutputTensor: Tensor2D | null = null;

// Module-local state for training data reference (for lookup)
let trainingInputArray: number[][] = [];
let trainingOutputArray: number[] = [];

// Module-local state for architecture display
let numLayers = 4;
let neuronsPerLayer = 6;

// Input format type and state
type InputFormat = 'number' | 'one-hot' | 'embedding';
let inputFormat: InputFormat = 'number';

// Function to set the input format (called from main.ts)
function setInputFormat(format: InputFormat): void {
  inputFormat = format;
  updateTextboxesFromInputs(vizInputArray);
}

// Format a single token number as a one-hot vector string
function formatTokenAsOneHot(tokenNum: number): string {
  const index = tokenNumberToIndex(tokenNum);
  const oneHot = Array(TOKENS.length).fill(0);
  oneHot[index] = 1;
  return '[' + oneHot.join(',') + ']';
}

// Format a single token number as an embedding vector string
function formatTokenAsEmbedding(tokenNum: number): string {
  const embedding = embedTokenNumber(tokenNum);
  return '[' + embedding.map(v => Number.isInteger(v) ? v.toString() : v.toFixed(2)).join(',') + ']';
}

// Initialize DOM elements by calling document.getElementById directly
function initVizDom() {
  if (domInitialized) return;
  outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  networkCanvas = document.getElementById('network-canvas') as HTMLCanvasElement;
  inputElements = Array.from({ length: VIZ_EXAMPLES_COUNT }, (_, i) => 
    document.getElementById(`input-${i}`) as HTMLInputElement
  ).filter((el): el is HTMLInputElement => el !== null);
  statusElement = document.getElementById('status') as HTMLElement;
  domInitialized = true;
}

function updateTextboxesFromInputs(inputArray: number[][]): void {
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = inputElements[i];
    if (inputElement && inputArray[i] !== undefined) {
      let formattedValue: string;
      switch (inputFormat) {
        case 'number':
          formattedValue = inputArray[i].map(tokenNumberToTokenString).join('');
          break;
        case 'one-hot':
          formattedValue = inputArray[i].map(formatTokenAsOneHot).join(' ');
          break;
        case 'embedding':
          formattedValue = inputArray[i].map(formatTokenAsEmbedding).join(' ');
          break;
      }
      inputElement.value = formattedValue;
    }
  }
}

function pickRandomInputs(): void {
  if (trainingInputArray.length === 0) return;
  
  const inputArray: number[][] = [];
  const outputArray: number[] = [];
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const randomIndex = Math.floor(Math.random() * trainingInputArray.length);
    inputArray.push(trainingInputArray[randomIndex]);
    outputArray.push(trainingOutputArray[randomIndex]);
  }

  const embeddedInputArray = inputArray.map(embedInput);

  // Dispose old tensors
  if (vizInputTensor) {
    try { vizInputTensor.dispose(); } catch (e) { /* ignore */ }
  }
  if (vizOutputTensor) {
    try { vizOutputTensor.dispose(); } catch (e) { /* ignore */ }
  }

  vizInputArray = inputArray;
  vizOutputArray = outputArray;
  vizInputTensor = tf.tensor2d(embeddedInputArray, [VIZ_EXAMPLES_COUNT, EMBEDDED_INPUT_SIZE]);
  vizOutputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  updateTextboxesFromInputs(inputArray);
}

function parseInputString(inputStr: string): number[] | null {
  const tokens: number[] = [];
  let i = 0;

  while (i < inputStr.length) {
    let matched = false;

    for (const token of TOKENS) {
      if (inputStr.substring(i, i + token.length) === token) {
        tokens.push(tokenStringToTokenNumber(token));
        i += token.length;
        matched = true;
        break;
      }
    }

    if (!matched) {
      return null;
    }
  }

  return tokens.length === INPUT_SIZE ? tokens : null;
}

function updateVizDataFromTextboxes(): void {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = inputElements[i];
    if (inputElement) {
      const parsed = parseInputString(inputElement.value);
      if (parsed) {
        inputArray.push(parsed);
        const matchingIndex = trainingInputArray.findIndex(arr =>
          arr.every((val, idx) => val === parsed[idx])
        );
        if (matchingIndex >= 0) {
          outputArray.push(trainingOutputArray[matchingIndex]);
        } else {
          outputArray.push(tokenStringToTokenNumber(NUMBERS[0]));
        }
      } else {
        if (vizInputArray[i]) {
          inputArray.push(vizInputArray[i]);
          outputArray.push(vizOutputArray[i]);
        } else if (trainingInputArray[0]) {
          inputArray.push(trainingInputArray[0]);
          outputArray.push(trainingOutputArray[0]);
        }
      }
    }
  }

  // Dispose old tensors
  if (vizInputTensor) {
    try { vizInputTensor.dispose(); } catch (e) { /* ignore */ }
  }
  if (vizOutputTensor) {
    try { vizOutputTensor.dispose(); } catch (e) { /* ignore */ }
  }

  const embeddedInputArray = inputArray.map(embedInput);
  vizInputArray = inputArray;
  vizOutputArray = outputArray;
  vizInputTensor = tf.tensor2d(embeddedInputArray, [VIZ_EXAMPLES_COUNT, EMBEDDED_INPUT_SIZE]);
  vizOutputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  drawViz();
}

async function drawViz(): Promise<void> {
  const model = getModel();
  if (!outputCanvas || !model || !vizInputTensor) return;
  
  const ctx = outputCanvas.getContext('2d')!;

  const predictionTensor = model.predict(vizInputTensor) as Tensor2D;
  const predictionArray = await predictionTensor.array() as number[][];

  ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

  const sectionSpacing = 10;
  const barSpacing = 3;

  const availableWidth = outputCanvas.width - (sectionSpacing * (VIZ_COLUMNS + 1));
  const sectionWidth = availableWidth / VIZ_COLUMNS;
  const availableHeight = outputCanvas.height - (sectionSpacing * (VIZ_ROWS + 1));
  const sectionHeight = availableHeight / VIZ_ROWS;

  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;

  for (let i = 0; i < vizInputArray.length; i++) {
    const col = i % VIZ_COLUMNS;
    const row = Math.floor(i / VIZ_COLUMNS);

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    const inputTokenStrings = vizInputArray[i].map(tokenNumberToTokenString).join('');
    ctx.font = '12px monospace';
    ctx.fillStyle = 'black';
    ctx.fillText(inputTokenStrings, sectionX + 5, sectionY + 15);

    const probabilities = predictionArray[i];
    const numBars = probabilities.length;
    const barWidth = (sectionWidth - barSpacing * (numBars + 1)) / numBars;

    for (let j = 0; j < probabilities.length; j++) {
      const barHeight = probabilities[j] * (sectionHeight - 40);
      const barX = sectionX + barSpacing + j * (barWidth + barSpacing);
      const barY = sectionY + sectionHeight - barHeight - barSpacing - 15;

      ctx.fillStyle = 'blue';
      ctx.fillRect(barX, barY, barWidth, barHeight);

      ctx.font = '10px monospace';
      ctx.fillStyle = 'black';
      ctx.fillText(indexToShortTokenString(j), barX, sectionY + sectionHeight - 5);
    }
  }

  predictionTensor.dispose();
}

function drawLossCurve(): void {
  const lossHistory = getLossHistory();
  if (!lossCanvas || lossHistory.length < 2) {
    return;
  }

  const ctx = lossCanvas.getContext('2d')!;
  ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

  const minLoss = Math.min(...lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...lossHistory.map(d => d.loss));
  const minEpoch = lossHistory[0].epoch;
  const maxEpoch = lossHistory[lossHistory.length - 1].epoch;

  const canvas = lossCanvas;
  function toCanvasX(epoch: number): number {
    return ((epoch - minEpoch) / (maxEpoch - minEpoch)) * (canvas.width - 60) + 30;
  }

  function toCanvasY(loss: number): number {
    const range = maxLoss - minLoss;
    const effectiveRange = range === 0 ? 1 : range;
    return canvas.height - 30 - ((loss - minLoss) / effectiveRange) * (canvas.height - 60);
  }

  ctx.strokeStyle = 'lightgrey';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toCanvasX(lossHistory[0].epoch), toCanvasY(lossHistory[0].loss));
  for (let i = 1; i < lossHistory.length; i++) {
    ctx.lineTo(toCanvasX(lossHistory[i].epoch), toCanvasY(lossHistory[i].loss));
  }
  ctx.stroke();
}

function drawNetworkArchitecture(): void {
  if (!networkCanvas) return;
  
  const ctx = networkCanvas.getContext('2d')!;
  ctx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

  const inputLayer = INPUT_SIZE;
  const embeddingLayer = EMBEDDED_INPUT_SIZE;
  const hiddenLayers = Array(numLayers).fill(neuronsPerLayer);
  const linearLayer = EMBEDDING_DIM;
  const outputLayer = OUTPUT_SIZE;
  const layers = [inputLayer, embeddingLayer, ...hiddenLayers, linearLayer, outputLayer];

  const layerHeight = 20;
  const maxLayerWidth = networkCanvas.width * 0.4;
  const layerGapY = 40;
  const startY = 30;
  const canvasWidth = networkCanvas.width;
  const arrowHeadSize = 8;

  const maxNeurons = Math.max(...layers);

  function drawDownwardArrow(ctx: CanvasRenderingContext2D, x: number, startY: number, endY: number): void {
    ctx.lineWidth = 6;
    ctx.strokeStyle = 'darkblue';
    ctx.fillStyle = 'darkblue';

    ctx.beginPath();
    ctx.moveTo(x, startY);
    ctx.lineTo(x, endY - arrowHeadSize);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, endY);
    ctx.lineTo(x - arrowHeadSize, endY - arrowHeadSize);
    ctx.lineTo(x + arrowHeadSize, endY - arrowHeadSize);
    ctx.closePath();
    ctx.fill();
  }

  const layerGeometries: { x: number; y: number; width: number; height: number }[] = [];

  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const layerWidth = (numNeurons / maxNeurons) * maxLayerWidth;
    const layerX = (canvasWidth / 2) - (layerWidth / 2);
    const layerY = startY + i * layerGapY;
    layerGeometries.push({ x: layerX, y: layerY, width: layerWidth, height: layerHeight });
  }

  ctx.strokeStyle = 'gray';
  ctx.lineWidth = 2;
  for (let i = 0; i < layerGeometries.length - 1; i++) {
    const currentLayer = layerGeometries[i];
    const nextLayer = layerGeometries[i + 1];
    const currentNeurons = layers[i];
    const nextNeurons = layers[i + 1];

    const currentNeuronPositions: { x: number; y: number }[] = [];
    for (let n = 0; n < currentNeurons; n++) {
      const x = currentLayer.x + (currentLayer.width / currentNeurons) * (n + 0.5);
      const y = currentLayer.y + currentLayer.height + 1;
      currentNeuronPositions.push({ x, y });
    }

    const nextNeuronPositions: { x: number; y: number }[] = [];
    for (let n = 0; n < nextNeurons; n++) {
      const x = nextLayer.x + (nextLayer.width / nextNeurons) * (n + 0.5);
      const y = nextLayer.y;
      nextNeuronPositions.push({ x, y });
    }

    const smallerCount = Math.min(currentNeurons, nextNeurons);
    const largerCount = Math.max(currentNeurons, nextNeurons);
    const isCurrentSmaller = currentNeurons <= nextNeurons;
    const smallerPositions = isCurrentSmaller ? currentNeuronPositions : nextNeuronPositions;
    const largerPositions = isCurrentSmaller ? nextNeuronPositions : currentNeuronPositions;

    const leftoverCount = Math.ceil((largerCount - smallerCount) / 2);
    const leftoverLeft = leftoverCount;
    const leftoverRight = leftoverCount;

    for (let l = 0; l < leftoverLeft; l++) {
      ctx.beginPath();
      ctx.moveTo(smallerPositions[0].x, smallerPositions[0].y);
      ctx.lineTo(largerPositions[l].x, largerPositions[l].y);
      ctx.stroke();
    }

    const bothEvenOrOdd = (currentNeurons % 2) === (nextNeurons % 2);

    if (bothEvenOrOdd) {
      const pairCount = smallerCount - 1;
      for (let p = 0; p < pairCount; p++) {
        const small1 = p;
        const small2 = p + 1;
        const large1 = leftoverLeft + p;
        const large2 = leftoverLeft + p + 1;

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
      const zigCount = smallerCount - 1;
      for (let z = 0; z < zigCount; z++) {
        const small1 = z;
        const small2 = z + 1;
        const large = leftoverLeft + z;

        ctx.beginPath();
        ctx.moveTo(smallerPositions[small1].x, smallerPositions[small1].y);
        ctx.lineTo(largerPositions[large].x, largerPositions[large].y);
        ctx.lineTo(smallerPositions[small2].x, smallerPositions[small2].y);
        ctx.stroke();
      }
    }

    for (let r = 0; r < leftoverRight; r++) {
      ctx.beginPath();
      ctx.moveTo(smallerPositions[smallerCount - 1].x, smallerPositions[smallerCount - 1].y);
      ctx.lineTo(largerPositions[largerCount - leftoverRight + r].x, largerPositions[largerCount - leftoverRight + r].y);
      ctx.stroke();
    }
  }

  for (let i = 0; i < layers.length; i++) {
    const layerNeurons = layers[i];
    const geom = layerGeometries[i];

    if (i === 0) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'darkblue';
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height);
      ctx.stroke();

      const arrowX = geom.x + geom.width / 2;
      const arrowStartY = geom.y;
      const arrowEndY = geom.y + geom.height - 2;

      drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
    } else {
      ctx.fillStyle = 'darkblue';
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

      ctx.lineWidth = 4;
      if (i === 1) {
        ctx.strokeStyle = '#90EE90';
      } else if (i >= 2 && i < layers.length - 2) {
        ctx.strokeStyle = '#4682B4';
      } else if (i === layers.length - 2) {
        ctx.strokeStyle = '#DDA0DD';
      } else if (i === layers.length - 1) {
        ctx.strokeStyle = 'rgba(255, 165, 0, 1)';
      }
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height - 1);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
      ctx.stroke();
    }

    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    let label = '';
    if (i === 0) {
      label = `${layerNeurons}-wide input`;
    } else if (i === 1) {
      label = `${layerNeurons}-wide embedding layer`;
    } else if (i === layers.length - 2) {
      label = `${layerNeurons}-wide linear layer`;
    } else if (i === layers.length - 1) {
      label = `${layerNeurons}-wide unembedding+softmax layer`;
    } else {
      label = `${layerNeurons}-wide ReLU layer`;
    }

    ctx.fillText(label, canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }

  const geom = layerGeometries[layerGeometries.length - 1];
  const arrowX = geom.x + geom.width / 2;
  const arrowStartY = geom.y + layerHeight + 3;
  const arrowEndY = geom.y + layerHeight + 3 + geom.height - 2;

  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, newNumLayers, newNeuronsPerLayer) => {
  numLayers = newNumLayers;
  neuronsPerLayer = newNeuronsPerLayer;
  
  // Pick random visualization inputs
  pickRandomInputs();

  // Visualize the initial (untrained) state
  drawViz();

  // Redraw the architecture in case it changed
  drawNetworkArchitecture();
};

// Implementation for the refreshViz message handler
const refreshViz: RefreshVizHandler = (_schedule) => {
  drawViz();
  drawLossCurve();
};

// Implementation for onEpochCompleted message handler
const onEpochCompleted: OnEpochCompletedHandler = (_schedule, epoch, loss) => {
  if (statusElement) {
    statusElement.innerHTML = `Training... Epoch ${epoch} - Loss: ${loss.toFixed(4)}`;
  }
};

// Implementation for setTrainingData message handler
const setTrainingData: SetTrainingDataHandler = (_schedule, data) => {
  trainingInputArray = data.inputArray;
  trainingOutputArray = data.outputArray;
};

// Dispose viz tensors
function disposeVizData() {
  if (vizInputTensor) {
    try { vizInputTensor.dispose(); } catch (e) { /* ignore */ }
    vizInputTensor = null;
  }
  if (vizOutputTensor) {
    try { vizOutputTensor.dispose(); } catch (e) { /* ignore */ }
    vizOutputTensor = null;
  }
}

// Set status message
function setStatusMessage(message: string) {
  if (statusElement) {
    statusElement.innerHTML = message;
  }
}

// Setup event listeners for input textboxes
function setupInputEventListeners() {
  initVizDom(); // Ensure DOM is initialized
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = inputElements[i];
    if (inputElement) {
      inputElement.addEventListener('input', () => {
        updateVizDataFromTextboxes();
      });
    }
  }
}

export {
  initVizDom,
  pickRandomInputs,
  updateVizDataFromTextboxes,
  drawViz,
  drawLossCurve,
  drawNetworkArchitecture,
  VIZ_EXAMPLES_COUNT,
  reinitializeModel,
  refreshViz,
  onEpochCompleted,
  setTrainingData,
  disposeVizData,
  setStatusMessage,
  setupInputEventListeners,
  setInputFormat
};
export type { InputFormat };
