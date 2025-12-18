import { INPUT_SIZE, OUTPUT_SIZE } from "./constants.js";
import { EMBEDDED_INPUT_SIZE, EMBEDDING_DIM } from "./embeddings.js";
import { AppState, DomElements } from "./types.js";
import { drawDownwardArrow, clearCanvas } from "./canvas-utils.js";
import { CANVAS_CONFIG } from "./config.js";

/**
 * Layer colors for visualization
 */
const LAYER_COLORS = {
  embedding: '#90EE90',
  hidden: '#4682B4',
  linear: '#DDA0DD',
  output: 'rgba(255, 165, 0, 1)',
  default: 'darkblue'
};

/**
 * Layout constants for network visualization
 */
const LAYOUT = CANVAS_CONFIG.networkArchitecture;

/**
 * Draw connections between two layers
 */
function drawLayerConnections(
  ctx: CanvasRenderingContext2D,
  currentLayer: { x: number; y: number; width: number; height: number },
  nextLayer: { x: number; y: number; width: number; height: number },
  currentNeurons: number,
  nextNeurons: number
): void {
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

  // Draw left connections
  for (let l = 0; l < leftoverLeft; l++) {
    ctx.beginPath();
    ctx.moveTo(smallerPositions[0].x, smallerPositions[0].y);
    ctx.lineTo(largerPositions[l].x, largerPositions[l].y);
    ctx.stroke();
  }

  const bothEvenOrOdd = (currentNeurons % 2) === (nextNeurons % 2);

  if (bothEvenOrOdd) {
    // Draw paired connections
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
    // Draw zigzag connections
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

  // Draw right connections
  for (let r = 0; r < leftoverRight; r++) {
    ctx.beginPath();
    ctx.moveTo(smallerPositions[smallerCount - 1].x, smallerPositions[smallerCount - 1].y);
    ctx.lineTo(largerPositions[largerCount - leftoverRight + r].x, largerPositions[largerCount - leftoverRight + r].y);
    ctx.stroke();
  }
}

/**
 * Get the label for a layer based on its index
 */
function getLayerLabel(index: number, numNeurons: number, totalLayers: number): string {
  if (index === 0) {
    return `${numNeurons}-wide input`;
  } else if (index === 1) {
    return `${numNeurons}-wide embedding layer`;
  } else if (index === totalLayers - 2) {
    return `${numNeurons}-wide linear layer`;
  } else if (index === totalLayers - 1) {
    return `${numNeurons}-wide unembedding+softmax layer`;
  } else {
    return `${numNeurons}-wide ReLU layer`;
  }
}

/**
 * Get the color for a layer based on its index
 */
function getLayerColor(index: number, totalLayers: number): string {
  if (index === 1) {
    return LAYER_COLORS.embedding;
  } else if (index >= 2 && index < totalLayers - 2) {
    return LAYER_COLORS.hidden;
  } else if (index === totalLayers - 2) {
    return LAYER_COLORS.linear;
  } else if (index === totalLayers - 1) {
    return LAYER_COLORS.output;
  }
  return LAYER_COLORS.default;
}

/**
 * Draw the network architecture visualization
 */
function drawNetworkArchitecture(appState: AppState, dom: DomElements): void {
  const canvas = dom.networkCanvas;
  const ctx = canvas.getContext('2d')!;
  clearCanvas(canvas);

  const inputLayer = INPUT_SIZE;
  const embeddingLayer = EMBEDDED_INPUT_SIZE;
  const hiddenLayers = Array(appState.num_layers).fill(appState.neurons_per_layer);
  const linearLayer = EMBEDDING_DIM;
  const outputLayer = OUTPUT_SIZE;
  const layers = [inputLayer, embeddingLayer, ...hiddenLayers, linearLayer, outputLayer];

  const maxLayerWidth = canvas.width * LAYOUT.maxLayerWidthRatio;
  const canvasWidth = canvas.width;
  const maxNeurons = Math.max(...layers);

  // Calculate layer geometries
  const layerGeometries: { x: number; y: number; width: number; height: number }[] = [];
  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const layerWidth = (numNeurons / maxNeurons) * maxLayerWidth;
    const layerX = (canvasWidth / 2) - (layerWidth / 2);
    const layerY = LAYOUT.startY + i * LAYOUT.layerGapY;
    layerGeometries.push({ x: layerX, y: layerY, width: layerWidth, height: LAYOUT.layerHeight });
  }

  // Draw connections between layers
  ctx.strokeStyle = 'gray';
  ctx.lineWidth = LAYOUT.lineWidth.connections;
  for (let i = 0; i < layerGeometries.length - 1; i++) {
    drawLayerConnections(
      ctx,
      layerGeometries[i],
      layerGeometries[i + 1],
      layers[i],
      layers[i + 1]
    );
  }

  // Draw layers
  for (let i = 0; i < layers.length; i++) {
    const numNeurons = layers[i];
    const geom = layerGeometries[i];

    if (i === 0) {
      // Draw input layer
      ctx.lineWidth = LAYOUT.lineWidth.inputBorder;
      ctx.strokeStyle = LAYER_COLORS.default;
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height);
      ctx.stroke();

      const arrowX = geom.x + geom.width / 2;
      const arrowStartY = geom.y;
      const arrowEndY = geom.y + geom.height - 2;
      drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
    } else {
      // Draw regular layer
      ctx.fillStyle = LAYER_COLORS.default;
      ctx.fillRect(geom.x, geom.y, geom.width, geom.height);

      ctx.lineWidth = LAYOUT.lineWidth.layerBorder;
      ctx.strokeStyle = getLayerColor(i, layers.length);
      ctx.beginPath();
      ctx.moveTo(geom.x, geom.y + geom.height - 1);
      ctx.lineTo(geom.x + geom.width, geom.y + geom.height - 1);
      ctx.stroke();
    }

    // Draw layer label
    ctx.fillStyle = 'black';
    ctx.textAlign = 'right';
    const label = getLayerLabel(i, numNeurons, layers.length);
    ctx.fillText(label, canvasWidth - 20, geom.y + geom.height / 2 + 5);
  }

  // Draw final output arrow
  const geom = layerGeometries[layerGeometries.length - 1];
  const arrowX = geom.x + geom.width / 2;
  const arrowStartY = geom.y + LAYOUT.layerHeight + 3;
  const arrowEndY = geom.y + LAYOUT.layerHeight + 3 + geom.height - 2;
  drawDownwardArrow(ctx, arrowX, arrowStartY, arrowEndY);
}

export { drawNetworkArchitecture };
