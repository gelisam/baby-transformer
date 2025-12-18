import { indexToShortTokenString, tokenNumberToTokenString } from "../tokens.js";
import { tf, Tensor2D } from "../tf.js";
import { TrainingData, AppState, DomElements } from "../types.js";
import { VIZ_ROWS, VIZ_COLUMNS } from "./constants.js";

async function drawPredictions(appState: AppState, vizData: TrainingData, dom: DomElements): Promise<void> {
  const canvas = dom.outputCanvas;
  const ctx = canvas.getContext('2d')!;

  const inputArray = vizData.inputArray;
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

export { drawPredictions };
