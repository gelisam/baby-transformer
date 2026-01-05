import { Schedule } from "../messageLoop.js";
import { OnEpochCompletedHandler } from "../messages/onEpochCompleted.js";
import { RefreshVizHandler } from "../messages/refreshViz.js";
import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";
import { getLossHistory } from "./model.js";

let lossCanvas: HTMLCanvasElement | null = null;
let statusElement: HTMLElement | null = null;

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

const reinitializeModel: ReinitializeModelHandler = (_schedule, _numLayers, _neuronsPerLayer, _inputFormat) => {
  const canvas = getLossCanvas();
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
};

const refreshViz: RefreshVizHandler = (_schedule) => {
  drawLossCurve();
};

const onEpochCompleted: OnEpochCompletedHandler = (_schedule, epoch, loss) => {
  const status = getStatusElement();
  status.innerHTML = `Training... Epoch ${epoch} - Loss: ${loss.toFixed(4)}`;
};

function setStatusMessage(message: string) {
  const status = getStatusElement();
  status.innerHTML = message;
}

export {
  drawLossCurve,
  reinitializeModel,
  refreshViz,
  onEpochCompleted,
  setStatusMessage
};
