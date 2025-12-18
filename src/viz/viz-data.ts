import { INPUT_SIZE, OUTPUT_SIZE } from "../constants.js";
import {
  NUMBERS,
  TOKENS,
  tokenNumberToIndex,
  tokenNumberToTokenString,
  tokenStringToTokenNumber
} from "../tokens.js";
import { EMBEDDED_INPUT_SIZE, embedInput } from "../embeddings.js";
import { tf, Tensor2D } from "../tf.js";
import { TrainingData, AppState, DomElements } from "../types.js";
import { VIZ_EXAMPLES_COUNT } from "./constants.js";
import { drawPredictions } from "./draw-predictions.js";

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

  updateTextboxesFromInputs(inputArray, outputArray, dom);

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

  drawPredictions(appState, appState.vizData, dom);
}

export { pickRandomInputs, updateVizDataFromTextboxes };
