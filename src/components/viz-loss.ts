import { Schedule } from "../messageLoop.js";
import { InitHandler } from "../messages/init.js";
import { OnEpochCompletedHandler } from "../messages/onEpochCompleted.js";
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { getLossHistory } from "./model.js";

// Module-local state for DOM elements (initialized on first use)
let lossCanvas: HTMLCanvasElement | null = null;
let statusElement: HTMLElement | null = null;

// Getter functions that check and initialize DOM elements if needed
function getLossCanvas(): HTMLCanvasElement {
  if (!lossCanvas) {
    lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  }
  return lossCanvas;
}

function getStatusElement(): HTMLElement {
  if (!statusElement) {
    statusElement = document.getElementById('status') as HTMLElement;
  }
  return statusElement;
}

// Handler for the Init message - no event listeners needed for this component
const init: InitHandler = (_schedule) => {
  // DOM elements will be lazily initialized when first accessed
};

function drawLossCurve(): void {
  const canvas = getLossCanvas();
  const lossHistory = getLossHistory();
  if (lossHistory.length < 2) {
    return;
  }

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const minLoss = Math.min(...lossHistory.map(d => d.loss));
  const maxLoss = Math.max(...lossHistory.map(d => d.loss));
  const minEpoch = lossHistory[0].epoch;
  const maxEpoch = lossHistory[lossHistory.length - 1].epoch;

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
  const status = getStatusElement();
  status.innerHTML = `Training... Epoch ${epoch} - Loss: ${loss.toFixed(4)}`;
};

// Set status message
function setStatusMessage(message: string) {
  const status = getStatusElement();
  status.innerHTML = message;
}

export {
  init,
  drawLossCurve,
  refreshViz,
  onEpochCompleted,
  setStatusMessage
};
