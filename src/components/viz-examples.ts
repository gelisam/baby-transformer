import {
  INPUT_SIZE,
  OUTPUT_SIZE
} from "../constants.js";
import type { InputFormat } from "../constants.js";
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
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetTrainingDataHandler } from "../messages/setTrainingData.js";
import { getModel } from "./model.js";

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

// Module-local state for DOM elements (initialized on first use)
let outputCanvas: HTMLCanvasElement | null = null;
let inputElements: HTMLInputElement[] = [];
let domInitialized = false;

// Module-local state for visualization data
let vizInputArray: number[][] = [];
let vizOutputArray: number[] = [];
let vizInputTensor: Tensor2D | null = null;
let vizOutputTensor: Tensor2D | null = null;

// Module-local state for training data reference (for lookup)
let trainingInputArray: number[][] = [];
let trainingOutputArray: number[] = [];

// Initialize DOM elements by calling document.getElementById directly
function initVizExamplesDom() {
  if (domInitialized) return;
  outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  inputElements = Array.from({ length: VIZ_EXAMPLES_COUNT }, (_, i) => 
    document.getElementById(`input-${i}`) as HTMLInputElement
  ).filter((el): el is HTMLInputElement => el !== null);
  domInitialized = true;
}

function updateTextboxesFromInputs(inputArray: number[][]): void {
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = inputElements[i];
    if (inputElement) {
      const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
      inputElement.value = inputTokenStrings;
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

  // Convert token numbers (1-6) to token indices (0-5) for the model
  const inputIndicesArray = inputArray.map(input => input.map(tokenNumberToIndex));

  // Dispose old tensors
  if (vizInputTensor) {
    try { vizInputTensor.dispose(); } catch (e) { /* ignore */ }
  }
  if (vizOutputTensor) {
    try { vizOutputTensor.dispose(); } catch (e) { /* ignore */ }
  }

  vizInputArray = inputArray;
  vizOutputArray = outputArray;
  vizInputTensor = tf.tensor2d(inputIndicesArray, [VIZ_EXAMPLES_COUNT, INPUT_SIZE]);
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

  // Convert token numbers (1-6) to token indices (0-5) for the model
  const inputIndicesArray = inputArray.map(input => input.map(tokenNumberToIndex));
  vizInputArray = inputArray;
  vizOutputArray = outputArray;
  vizInputTensor = tf.tensor2d(inputIndicesArray, [VIZ_EXAMPLES_COUNT, INPUT_SIZE]);
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

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, _newNumLayers, _newNeuronsPerLayer, _newInputFormat) => {
  // Pick random visualization inputs
  pickRandomInputs();

  // Visualize the initial (untrained) state
  drawViz();
};

// Implementation for the refreshViz message handler
const refreshViz: RefreshVizHandler = (_schedule) => {
  drawViz();
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

// Setup event listeners for input textboxes
function setupInputEventListeners() {
  initVizExamplesDom(); // Ensure DOM is initialized
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
  initVizExamplesDom,
  pickRandomInputs,
  updateVizDataFromTextboxes,
  drawViz,
  VIZ_EXAMPLES_COUNT,
  reinitializeModel,
  refreshViz,
  setTrainingData,
  disposeVizData,
  setupInputEventListeners
};
