import { INPUT_SIZE } from "../constants.js";
import { EMBEDDED_INPUT_SIZE, EMBEDDING_DIM } from "../embeddings.js";
import { AppState, DomElements } from "../types.js";
import {
  NETWORK_LAYER_HEIGHT,
  NETWORK_LAYER_GAP_Y,
  NETWORK_START_Y,
  NETWORK_MAX_WIDTH_RATIO,
  NETWORK_ARROW_HEAD_SIZE,
  LAYER_COLORS
} from "./constants.js";
import { OUTPUT_SIZE } from "../constants.js";

interface LayerGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
}

function drawDownwardArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  startY: number,
  endY: number
): void {
  ctx.lineWidth = 6;
  ctx.strokeStyle = LAYER_COLORS.primary;
  ctx.fillStyle = LAYER_COLORS.primary;

  ctx.beginPath();
  ctx.moveTo(x, startY);
  ctx.lineTo(x, endY - NETWORK_ARROW_HEAD_SIZE);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x, endY);
  ctx.lineTo(x - NETWORK_ARROW_HEAD_SIZE, endY - NETWORK_ARROW_HEAD_SIZE);
  ctx.lineTo(x + NETWORK_ARROW_HEAD_SIZE, endY - NETWORK_ARROW_HEAD_SIZE);
  ctx.closePath();
  ctx.fill();
}

function calculateLayerGeometries(
  layers: number[],
  canvasWidth: number,
  maxLayerWidth: number
): LayerGeometry[] {
  const maxNeurons = Math.max(...layers);
  const geometries: LayerGeometry[] = [];

  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const layerWidth = (numNeurons / maxNeurons) * maxLayerWidth;
    const layerX = (canvasWidth / 2) - (layerWidth / 2);
    const layerY = NETWORK_START_Y + i * NETWORK_LAYER_GAP_Y;
    geometries.push({ x: layerX, y: layerY, width: layerWidth, height: NETWORK_LAYER_HEIGHT });
  }

  return geometries;
}

function drawLayerConnections(
  ctx: CanvasRenderingContext2D,
  layers: number[],
  layerGeometries: LayerGeometry[]
): void {
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

    drawConnectionsBetweenLayers(
      ctx,
      currentNeuronPositions,
      nextNeuronPositions,
      currentNeurons,
      nextNeurons
    );
  }
}

function drawConnectionsBetweenLayers(
  ctx: CanvasRenderingContext2D,
  currentPositions: { x: number; y: number }[],
  nextPositions: { x: number; y: number }[],
  currentNeurons: number,
  nextNeurons: number
): void {
  const smallerCount = Math.min(currentNeurons, nextNeurons);
  const largerCount = Math.max(currentNeurons, nextNeurons);
  const isCurrentSmaller = currentNeurons <= nextNeurons;
  const smallerPositions = isCurrentSmaller ? currentPositions : nextPositions;
  const largerPositions = isCurrentSmaller ? nextPositions : currentPositions;

  const leftoverCount = Math.ceil((largerCount - smallerCount) / 2);

  // Draw left overflow connections
  for (let l = 0; l < leftoverCount; l++) {
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
      const large1 = leftoverCount + p;
      const large2 = leftoverCount + p + 1;

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
      const large = leftoverCount + z;

      ctx.beginPath();
      ctx.moveTo(smallerPositions[small1].x, smallerPositions[small1].y);
      ctx.lineTo(largerPositions[large].x, largerPositions[large].y);
      ctx.lineTo(smallerPositions[small2].x, smallerPositions[small2].y);
      ctx.stroke();
    }
  }

  // Draw right overflow connections
  for (let r = 0; r < leftoverCount; r++) {
    ctx.beginPath();
    ctx.moveTo(smallerPositions[smallerCount - 1].x, smallerPositions[smallerCount - 1].y);
    ctx.lineTo(largerPositions[largerCount - leftoverCount + r].x, largerPositions[largerCount - leftoverCount + r].y);
    ctx.stroke();
  }
}

function getLayerColor(layerIndex: number, totalLayers: number): string {
  if (layerIndex === 1) {
    return LAYER_COLORS.embedding;
  } else if (layerIndex >= 2 && layerIndex < totalLayers - 2) {
    return LAYER_COLORS.hidden;
  } else if (layerIndex === totalLayers - 2) {
    return LAYER_COLORS.linear;
  } else if (layerIndex === totalLayers - 1) {
    return LAYER_COLORS.output;
  }
  return LAYER_COLORS.primary;
}

function getLayerLabel(layerIndex: number, numNeurons: number, totalLayers: number): string {
  if (layerIndex === 0) {
    return `${numNeurons}-wide input`;
  } else if (layerIndex === 1) {
    return `${numNeurons}-wide embedding layer`;
  } else if (layerIndex === totalLayers - 2) {
    return `${numNeurons}-wide linear layer`;
  } else if (layerIndex === totalLayers - 1) {
    return `${numNeurons}-wide unembedding+softmax layer`;
  }
  return `${numNeurons}-wide ReLU layer`;
}

function drawLayers(
  ctx: CanvasRenderingContext2D,
  layers: number[],
  layerGeometries: LayerGeometry[],
  canvasWidth: number
): void {
  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const geom = layerGeometries[i];

    if (i === 0) {
      // Input layer - draw as a line with arrow
      ctx.lineWidth = 2;
      ctx.strokeStyle = LAYER_COLORS.primary;
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height);
      ctx.stroke();

      const arrowX = geom.x + geom.width / 2;
      const arrowStartY = geom.y;
      const arrowEndY = geom.y + geom.height - 2;
      drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
    } else {
      // Hidden and output layers - draw as filled rectangles
      ctx.fillStyle = LAYER_COLORS.primary;
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

      // Draw colored underline
      ctx.lineWidth = 4;
      ctx.strokeStyle = getLayerColor(i, layers.length);
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height - 1);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
      ctx.stroke();
    }

    // Draw label
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    ctx.fillText(getLayerLabel(i, numNeurons, layers.length), canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }
}

function drawNetworkArchitecture(appState: AppState, dom: DomElements): void {
  const canvas = dom.networkCanvas;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const inputLayer = INPUT_SIZE;
  const embeddingLayer = EMBEDDED_INPUT_SIZE;
  const hiddenLayers = Array(appState.numLayers).fill(appState.neuronsPerLayer);
  const linearLayer = EMBEDDING_DIM;
  const outputLayer = OUTPUT_SIZE;
  const layers = [inputLayer, embeddingLayer, ...hiddenLayers, linearLayer, outputLayer];

  const maxLayerWidth = canvas.width * NETWORK_MAX_WIDTH_RATIO;
  const layerGeometries = calculateLayerGeometries(layers, canvas.width, maxLayerWidth);

  // Draw connections between layers
  drawLayerConnections(ctx, layers, layerGeometries);

  // Draw the layers themselves
  drawLayers(ctx, layers, layerGeometries, canvas.width);

  // Draw output arrow
  const lastGeom = layerGeometries[layerGeometries.length - 1];
  const arrowX = lastGeom.x + lastGeom.width / 2;
  const arrowStartY = lastGeom.y + NETWORK_LAYER_HEIGHT + 3;
  const arrowEndY = lastGeom.y + NETWORK_LAYER_HEIGHT + 3 + lastGeom.height - 2;
  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}

export { drawNetworkArchitecture };
