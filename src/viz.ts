import { INPUT_SIZE, OUTPUT_SIZE } from "./constants.js";
import { NUMBERS, indexToShortTokenString, tokenNumberToTokenString, tokenStringToTokenNumber } from "./tokens.js";
import { Tensor2D } from "./tf.js";
import { TrainingData, AppState, DomElements } from "./types.js";
import { parseInputString } from "./input-parser.js";
import { createTrainingData, pickRandomExamples } from "./data-manager.js";
import { clearCanvas } from "./canvas-utils.js";
import { ResourceManager } from "./resource-manager.js";
import { VIZ_CONFIG, CANVAS_CONFIG } from "./config.js";

const VIZ_ROWS = VIZ_CONFIG.rows;
const VIZ_COLUMNS = VIZ_CONFIG.columns;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

const resourceManager = new ResourceManager();

function updateTextboxesFromInputs(inputArray: number[][], outputArray: number[], dom: DomElements): void {
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = dom.inputElements[i];
    if (inputElement) {
      const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
      inputElement.value = inputTokenStrings;
    }
  }
}

function pickRandomInputs(data: TrainingData, dom: DomElements): TrainingData {
  const { inputArray, outputArray } = pickRandomExamples(data, VIZ_EXAMPLES_COUNT);
  const vizData = createTrainingData(inputArray, outputArray);
  updateTextboxesFromInputs(inputArray, outputArray, dom);
  return vizData;
}

function updateVizDataFromTextboxes(appState: AppState, dom: DomElements): void {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = dom.inputElements[i];
    if (inputElement) {
      const parsed = parseInputString(inputElement.value);
      if (parsed) {
        inputArray.push(parsed);
        const matchingIndex = appState.data.inputArray.findIndex(arr =>
          arr.every((val, idx) => val === parsed[idx])
        );
        if (matchingIndex >= 0) {
          outputArray.push(appState.data.outputArray[matchingIndex]);
        } else {
          outputArray.push(tokenStringToTokenNumber(NUMBERS[0]));
        }
      } else {
        if (appState.vizData && appState.vizData.inputArray[i]) {
          inputArray.push(appState.vizData.inputArray[i]);
          outputArray.push(appState.vizData.outputArray[i]);
        } else {
          inputArray.push(appState.data.inputArray[0]);
          outputArray.push(appState.data.outputArray[0]);
        }
      }
    }
  }

  // Use ResourceManager for safe disposal
  resourceManager.disposeTrainingData(appState.vizData);

  appState.vizData = createTrainingData(inputArray, outputArray);
  drawViz(appState, appState.vizData, dom);
}

async function drawViz(appState: AppState, vizData: TrainingData, dom: DomElements): Promise<void> {
  const canvas = dom.outputCanvas;
  const ctx = canvas.getContext('2d')!;

  const inputArray = vizData.inputArray;
  const outputArray = vizData.outputArray;
  const inputTensor = vizData.inputTensor;

  const predictionTensor = appState.model.predict(inputTensor) as Tensor2D;
  const predictionArray = await predictionTensor.array() as number[][];

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const sectionSpacing = VIZ_CONFIG.sectionSpacing;
  const barSpacing = VIZ_CONFIG.barSpacing;

  const availableWidth = canvas.width - (sectionSpacing * (VIZ_COLUMNS + 1));
  const sectionWidth = availableWidth / VIZ_COLUMNS;
  const availableHeight = canvas.height - (sectionSpacing * (VIZ_ROWS + 1));
  const sectionHeight = availableHeight / VIZ_ROWS;

  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;

  for (let i = 0; i < inputArray.length; i++) {
    const col = i % VIZ_COLUMNS;
    const row = Math.floor(i / VIZ_COLUMNS);

    const sectionX = sectionSpacing + col * (sectionWidth + sectionSpacing);
    const sectionY = sectionSpacing + row * (sectionHeight + sectionSpacing);

    ctx.strokeRect(sectionX, sectionY, sectionWidth, sectionHeight);

    const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
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

function drawLossCurve(appState: AppState, dom: DomElements): void {
  if (appState.lossHistory.length < 2) {
    return;
  }

  const canvas = dom.lossCanvas;
  const ctx = canvas.getContext('2d')!;
  clearCanvas(canvas);

  const minLoss = Math.min(...appState.lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...appState.lossHistory.map(d => d.loss));
  const minEpoch = appState.lossHistory[0].epoch;
  const maxEpoch = appState.lossHistory[appState.lossHistory.length - 1].epoch;

  const margin = CANVAS_CONFIG.lossChart;

  function toCanvasX(epoch: number): number {
    return ((epoch - minEpoch) / (maxEpoch - minEpoch)) * (canvas.width - 2 * margin.marginX) + margin.marginX;
  }

  function toCanvasY(loss: number): number {
    const range = maxLoss - minLoss;
    const effectiveRange = range === 0 ? 1 : range;
    return canvas.height - margin.marginY - ((loss - minLoss) / effectiveRange) * (canvas.height - 2 * margin.marginY);
  }

  ctx.strokeStyle = 'lightgrey';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toCanvasX(appState.lossHistory[0].epoch), toCanvasY(appState.lossHistory[0].loss));
  for (let i = 1; i < appState.lossHistory.length; i++) {
    ctx.lineTo(toCanvasX(appState.lossHistory[i].epoch), toCanvasY(appState.lossHistory[i].loss));
  }
  ctx.stroke();
}

export {
  pickRandomInputs,
  updateVizDataFromTextboxes,
  drawViz,
  drawLossCurve,
  VIZ_EXAMPLES_COUNT
};
