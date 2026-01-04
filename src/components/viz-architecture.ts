import {
  INPUT_SIZE,
  EMBEDDING_DIM,
  getOutputSize,
  getTransformedInputSize
} from "../constants.js";
import type { InputFormat } from "../constants.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";

// Module-local state for DOM elements (initialized on first use)
let networkCanvas: HTMLCanvasElement | null = null;

// Module-local state for architecture display
let numLayers = 4;
let neuronsPerLayer = 6;
let currentInputFormat: InputFormat = 'embedding';
let currentVocabSize = 3;

// Getter function that checks and initializes DOM element if needed
function getNetworkCanvas(): HTMLCanvasElement {
  if (!networkCanvas) {
    networkCanvas = document.getElementById('network-canvas') as HTMLCanvasElement;
  }
  return networkCanvas;
}

function drawNetworkArchitecture(): void {
  const canvas = getNetworkCanvas();
  
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const transformedInputSize = getTransformedInputSize(currentInputFormat, currentVocabSize);
  
  // Build layers array based on current input format
  // For embedding format: input -> embedding preprocessing -> ReLU layers -> linear -> softmax
  // For one-hot format: input -> one-hot preprocessing -> ReLU layers -> linear -> softmax
  // For number format: input -> ReLU layers -> linear -> softmax (no preprocessing layer shown)
  const inputLayer = INPUT_SIZE;
  const hiddenLayers = Array(numLayers).fill(neuronsPerLayer);
  const linearLayer = EMBEDDING_DIM;
  const outputLayer = getOutputSize(currentVocabSize);
  
  // Build layers with optional preprocessing layer
  let layers: number[];
  let layerLabels: string[];
  
  if (currentInputFormat === 'embedding') {
    layers = [inputLayer, transformedInputSize, ...hiddenLayers, linearLayer, outputLayer];
    layerLabels = [
      `${inputLayer}-wide input`,
      `${transformedInputSize}-wide embedding layer`,
      ...hiddenLayers.map(n => `${n}-wide ReLU layer`),
      `${linearLayer}-wide linear layer`,
      `${outputLayer}-wide unembedding+softmax layer`
    ];
  } else if (currentInputFormat === 'one-hot') {
    layers = [inputLayer, transformedInputSize, ...hiddenLayers, linearLayer, outputLayer];
    layerLabels = [
      `${inputLayer}-wide input`,
      `${transformedInputSize}-wide one-hot layer`,
      ...hiddenLayers.map(n => `${n}-wide ReLU layer`),
      `${linearLayer}-wide linear layer`,
      `${outputLayer}-wide unembedding+softmax layer`
    ];
  } else {
    // number format - no preprocessing layer
    layers = [inputLayer, ...hiddenLayers, linearLayer, outputLayer];
    layerLabels = [
      `${inputLayer}-wide input`,
      ...hiddenLayers.map(n => `${n}-wide ReLU layer`),
      `${linearLayer}-wide linear layer`,
      `${outputLayer}-wide unembedding+softmax layer`
    ];
  }

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

  // Determine which layer index is the preprocessing layer (if any)
  const hasPreprocessingLayer = currentInputFormat !== 'number';
  const preprocessingLayerIndex = hasPreprocessingLayer ? 1 : -1;
  const firstReluIndex = hasPreprocessingLayer ? 2 : 1;
  const linearLayerIndex = layers.length - 2;
  const softmaxLayerIndex = layers.length - 1;

  for (let i = 0; i < layers.length; i++) {
    const geom = layerGeometries[i];

    if (i === 0) {
      // Input layer - draw as line with arrow
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
    } else if (i === preprocessingLayerIndex && currentInputFormat === 'one-hot') {
      // One-hot layer: white rectangle with thin vertical blue bars per input
      ctx.fillStyle = 'white';
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      ctx.strokeRect(geom.x, geom.y, geom.width, geom.height);

      // Draw thin vertical bars per input, occupying the bottom 25% of the layer height
      const numInputs = INPUT_SIZE;
      const barWidth = 3;
      const barHeight = geom.height * 0.25;
      ctx.fillStyle = 'blue';
      for (let b = 0; b < numInputs; b++) {
        const barX = geom.x + (geom.width / numInputs) * (b + 0.5) - barWidth / 2;
        const barY = geom.y + geom.height - barHeight;
        ctx.fillRect(barX, barY, barWidth, barHeight);
      }
    } else {
      ctx.fillStyle = 'darkblue';
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

      ctx.lineWidth = 4;
      if (i === preprocessingLayerIndex) {
        ctx.strokeStyle = '#90EE90'; // Light green for preprocessing
      } else if (i >= firstReluIndex && i < linearLayerIndex) {
        ctx.strokeStyle = '#4682B4'; // Steel blue for ReLU
      } else if (i === linearLayerIndex) {
        ctx.strokeStyle = '#DDA0DD'; // Plum for linear
      } else if (i === softmaxLayerIndex) {
        ctx.strokeStyle = 'rgba(255, 165, 0, 1)'; // Orange for softmax
      }
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height - 1);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
      ctx.stroke();
    }

    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.fillText(layerLabels[i], canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }

  const geom = layerGeometries[layerGeometries.length - 1];
  const arrowX = geom.x + geom.width / 2;
  const arrowStartY = geom.y + layerHeight + 3;
  const arrowEndY = geom.y + layerHeight + 3 + geom.height - 2;

  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}

// Implementation for the reinitializeModel message handler
const reinitializeModel: ReinitializeModelHandler = (_schedule, newNumLayers, newNeuronsPerLayer, newInputFormat, vocabSize) => {
  numLayers = newNumLayers;
  neuronsPerLayer = newNeuronsPerLayer;
  currentInputFormat = newInputFormat;
  currentVocabSize = vocabSize;
  
  // Redraw the architecture in case it changed
  drawNetworkArchitecture();
};

export {
  drawNetworkArchitecture,
  reinitializeModel
};
