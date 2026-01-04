import {
  INPUT_SIZE,
} from "../constants.js";
import type { InputFormat } from "../constants.js";
import { tf, Tensor2D } from "../tf.js";
import {
  getTokenCount,
  getTokenAtIndex,
  getNumberAtIndex,
  indexToShortTokenString,
  tokenNumberToIndex,
  tokenNumberToTokenString,
  tokenStringToTokenNumber
} from "../tokens.js";
import { Schedule } from "../messageLoop.js";
import { InitHandler } from "../messages/init.js";
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { SetTrainingDataHandler } from "../messages/setTrainingData.js";
import { getModel } from "./model.js";

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

// Module-local state for DOM elements (initialized on first use)
let outputCanvas: HTMLCanvasElement | null = null;
let inputElements: HTMLInputElement[] | null = null;

// Module-local state for visualization data
let vizInputArray: number[][] = [];
let vizOutputArray: number[] = [];
let vizInputTensor: Tensor2D | null = null;
let vizOutputTensor: Tensor2D | null = null;

// Module-local state for training data reference (for lookup)
let trainingInputArray: number[][] = [];
let trainingOutputArray: number[] = [];

// Module-local state for current vocabulary size
let currentVocabSize = 3;

// Getter functions that check and initialize DOM elements if needed
function getOutputCanvas(): HTMLCanvasElement {
  if (!outputCanvas) {
    outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  }
  return outputCanvas;
}

function getInputElements(): HTMLInputElement[] {
  if (!inputElements) {
    inputElements = Array.from({ length: VIZ_EXAMPLES_COUNT }, (_, i) => 
      document.getElementById(`input-${i}`) as HTMLInputElement
    ).filter((el): el is HTMLInputElement => el !== null);
  }
  return inputElements;
}

// Handler for the Init message - attach event listeners for input textboxes
const init: InitHandler = (_schedule) => {
  const elements = getInputElements();
  for (let i = 0; i < elements.length; i++) {
    const inputElement = elements[i];
    inputElement.addEventListener('input', () => {
      updateVizDataFromTextboxes();
    });
  }
};

function updateTextboxesFromInputs(inputArray: number[][]): void {
  const elements = getInputElements();
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = elements[i];
    if (inputElement) {
      const inputTokenStrings = inputArray[i].map(t => tokenNumberToTokenString(currentVocabSize, t)).join('');
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
  const outputSize = currentVocabSize * 2;
  vizOutputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), outputSize) as Tensor2D;

  updateTextboxesFromInputs(inputArray);
}

function parseInputString(inputStr: string): number[] | null {
  const tokens: number[] = [];
  let i = 0;
  const tokenCount = getTokenCount(currentVocabSize);

  while (i < inputStr.length) {
    let matched = false;

    for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++) {
      const token = getTokenAtIndex(currentVocabSize, tokenIndex);
      if (inputStr.substring(i, i + token.length) === token) {
        tokens.push(tokenStringToTokenNumber(currentVocabSize, token));
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
  const elements = getInputElements();
  const inputArray: number[][] = [];
  const outputArray: number[] = [];
  const firstNumber = getNumberAtIndex(currentVocabSize, 0);

  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = elements[i];
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
          outputArray.push(tokenStringToTokenNumber(currentVocabSize, firstNumber));
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
  const outputSize = currentVocabSize * 2;
  vizOutputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), outputSize) as Tensor2D;

  drawViz();
}

async function drawViz(): Promise<void> {
  const model = getModel();
  if (!model || !vizInputTensor) return;
  
  const canvas = getOutputCanvas();
  const ctx = canvas.getContext('2d')!;

  const predictionTensor = model.predict(vizInputTensor) as Tensor2D;
  const predictionArray = await predictionTensor.array() as number[][];

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const sectionSpacing = 10;
  const barSpacing = 3;

  const availableWidth = canvas.width - (sectionSpacing * (VIZ_COLUMNS + 1));
  const sectionWidth = availableWidth / VIZ_COLUMNS;
  const availableHeight = canvas.height - (sectionSpacing * (VIZ_ROWS + 1));
  const sectionHeight = availableHeight / VIZ_ROWS;

  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;

  for (let i = 0; i < vizInputArray.length; i++) {
    const col = i % VIZ_COLUMNS;
    const row = Math.floor(i / VIZ_COLUMNS);

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    const inputTokenStrings = vizInputArray[i].map(t => tokenNumberToTokenString(currentVocabSize, t)).join('');
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
      ctx.fillText(indexToShortTokenString(currentVocabSize, j), barX, sectionY + sectionHeight - 5);
    }
  }

  predictionTensor.dispose();
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, _newNumLayers, _newNeuronsPerLayer, _newInputFormat, vocabSize) => {
  currentVocabSize = vocabSize;
  
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

export {
  init,
  pickRandomInputs,
  updateVizDataFromTextboxes,
  drawViz,
  VIZ_EXAMPLES_COUNT,
  reinitializeModel,
  refreshViz,
  setTrainingData,
  disposeVizData
};
