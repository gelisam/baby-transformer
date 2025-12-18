import { AppState, DomElements } from "../types.js";

function drawLossCurve(appState: AppState, dom: DomElements): void {
  if (appState.lossHistory.length < 2) {
    return;
  }

  const canvas = dom.lossCanvas;
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

export { drawLossCurve };
