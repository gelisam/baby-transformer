import { Schedule } from "../messageLoop.js";
import { OnEpochCompletedHandler } from "../messages/onEpochCompleted.js";
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { getLossHistory } from "./model.js";

// Module-local state for DOM elements (initialized on first use)
let lossCanvas: HTMLCanvasElement | null = null;
let statusElement: HTMLElement | null = null;
let domInitialized = false;

// Initialize DOM elements by calling document.getElementById directly
function initVizLossDom() {
  if (domInitialized) return;
  lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  statusElement = document.getElementById('status') as HTMLElement;
  domInitialized = true;
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

// Implementation for the refreshViz message handler
const refreshViz: RefreshVizHandler = (_schedule) => {
  drawLossCurve();
};

// Implementation for onEpochCompleted message handler
const onEpochCompleted: OnEpochCompletedHandler = (_schedule, epoch, loss) => {
  if (statusElement) {
    statusElement.innerHTML = `Training... Epoch ${epoch} - Loss: ${loss.toFixed(4)}`;
  }
};

// Set status message
function setStatusMessage(message: string) {
  if (statusElement) {
    statusElement.innerHTML = message;
  }
}

export {
  initVizLossDom,
  drawLossCurve,
  refreshViz,
  onEpochCompleted,
  setStatusMessage
};
