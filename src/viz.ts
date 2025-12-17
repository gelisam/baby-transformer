import {
  INPUT_SIZE,
  OUTPUT_SIZE
} from "./constants.js";
import {
  NUMBERS,
  TOKENS,
  indexToShortTokenString,
  tokenNumberToIndex,
  tokenNumberToTokenString,
  tokenStringToTokenNumber
} from "./tokens.js";
import {
  EMBEDDED_INPUT_SIZE,
  EMBEDDING_DIM,
  embedInput
} from "./embeddings.js";
import { tf, Tensor2D } from "./tf.js";
import { TrainingData, AppState } from "./types.js";

const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

function updateTextboxesFromInputs(inputArray: number[][], outputArray: number[]): void {
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = document.getElementById(`input-${i}`) as HTMLInputElement;
    if (inputElement) {
      const inputTokenStrings = inputArray[i].map(tokenNumberToTokenString).join('');
      inputElement.value = inputTokenStrings;
    }
  }
}

function pickRandomInputs(data: TrainingData): TrainingData {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];
  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const randomIndex = Math.floor(Math.random() * data.inputArray.length);
    inputArray.push(data.inputArray[randomIndex]);
    outputArray.push(data.outputArray[randomIndex]);
  }

  const embeddedInputArray = inputArray.map(embedInput);

  const inputTensor = tf.tensor2d(embeddedInputArray, [VIZ_EXAMPLES_COUNT, EMBEDDED_INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  updateTextboxesFromInputs(inputArray, outputArray);

  return {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };
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

function updateVizDataFromTextboxes(appState: AppState): void {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  for (let i = 0; i < VIZ_EXAMPLES_COUNT; i++) {
    const inputElement = document.getElementById(`input-${i}`) as HTMLInputElement;
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

  if (appState.vizData) {
    appState.vizData.inputTensor.dispose();
    appState.vizData.outputTensor.dispose();
  }

  const embeddedInputArray = inputArray.map(embedInput);
  const inputTensor = tf.tensor2d(embeddedInputArray, [VIZ_EXAMPLES_COUNT, EMBEDDED_INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  appState.vizData = {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };

  drawViz(appState, appState.vizData);
}

async function drawViz(appState: AppState, vizData: TrainingData): Promise<void> {
  const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;

  const inputArray = vizData.inputArray;
  const outputArray = vizData.outputArray;
  const inputTensor = vizData.inputTensor;

  const predictionTensor = appState.model.predict(inputTensor) as Tensor2D;
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

function drawLossCurve(appState: AppState): void {
  if (appState.lossHistory.length < 2) {
    return;
  }

  const canvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const minLoss = Math.min(...appState.lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...appState.lossHistory.map(d => d.loss));
  const minEpoch = appState.lossHistory[0].epoch;
  const maxEpoch = appState.lossHistory[appState.lossHistory.length - 1].epoch;

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
  ctx.moveTo(toCanvasX(appState.lossHistory[0].epoch), toCanvasY(appState.lossHistory[0].loss));
  for (let i = 1; i < appState.lossHistory.length; i++) {
    ctx.lineTo(toCanvasX(appState.lossHistory[i].epoch), toCanvasY(appState.lossHistory[i].loss));
  }
  ctx.stroke();
}

function drawNetworkArchitecture(appState: AppState): void {
  const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const inputLayer = INPUT_SIZE;
  const embeddingLayer = EMBEDDED_INPUT_SIZE;
  const hiddenLayers = Array(appState.num_layers).fill(appState.neurons_per_layer);
  const linearLayer = EMBEDDING_DIM;
  const outputLayer = OUTPUT_SIZE;
  const layers = [inputLayer, embeddingLayer, ...hiddenLayers, linearLayer, outputLayer];

  const layerHeight = 20;
  const maxLayerWidth = canvas.width * 0.4;
  const layerGapY = 40;
  const startY = 30;
  const canvasWidth = canvas.width;
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
    const numNeurons = layers[i];
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
      label = `${numNeurons}-wide input`;
    } else if (i === 1) {
      label = `${numNeurons}-wide embedding layer`;
    } else if (i === layers.length - 2) {
      label = `${numNeurons}-wide linear layer`;
    } else if (i === layers.length - 1) {
      label = `${numNeurons}-wide unembedding+softmax layer`;
    } else {
      label = `${numNeurons}-wide ReLU layer`;
    }

    ctx.fillText(label, canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }

  const geom = layerGeometries[layerGeometries.length - 1];
  const arrowX = geom.x + geom.width / 2;
  const arrowStartY = geom.y + layerHeight + 3;
  const arrowEndY = geom.y + layerHeight + 3 + geom.height - 2;

  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}

export {
  pickRandomInputs,
  updateVizDataFromTextboxes,
  drawViz,
  drawLossCurve,
  drawNetworkArchitecture,
  VIZ_ROWS,
  VIZ_COLUMNS,
  VIZ_EXAMPLES_COUNT
};
