import {
  NUMBERS,
  LETTERS,
  tokenNumberToIndex,
  tokenStringToTokenNumber
} from "./tokens.js";
import {
  INPUT_SIZE,
  OUTPUT_SIZE,
} from "./constants.js";
import { EMBEDDED_INPUT_SIZE, embedInput } from "./embeddings.js";
import { tf, Tensor2D } from "./tf.js";

// Module-local state for training data
let trainingInputArray: number[][] = [];
let trainingOutputArray: number[] = [];
let trainingInputTensor: Tensor2D | null = null;
let trainingOutputTensor: Tensor2D | null = null;

// Generate training data for the classification task
function generateData(): void {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];
  function addExample(sequence: number[]) {
    //console.log(sequence.map(tokenNumberToTokenString).join(''));
    const input = sequence.slice(0, INPUT_SIZE);
    const output = sequence[INPUT_SIZE];
    inputArray.push(input);
    outputArray.push(output);
  }

  function removeElementAt(arr: number[], index: number): number[] {
    const newArr = arr.slice(); // ["a", "b", "c"]
    newArr.splice(index, 1); // returns ["b"] and mutates 'newArr' to ["a", "c"]
    return newArr;
  }

  function insert(mapping: Map<number, number>, key: number, value: number): Map<number, number> {
    const newMapping = new Map(mapping);
    newMapping.set(key, value);
    return newMapping;
  }

  const allLetters = LETTERS.map(tokenStringToTokenNumber);
  const allNumbers = NUMBERS.map(tokenStringToTokenNumber);
  function generate(
      n: number, // number of examples to generate before the final pair
      sequence: number[],
      mapping: Map<number, number>,
      availableLetters: number[],
      availableNumbers: number[]
  ) {
    if (n === 0) {
      for (const letter of allLetters) {
        if (mapping.has(letter)) {
          addExample([...sequence, letter, mapping.get(letter)!]);
        } else {
          for (const number of availableNumbers) {
            addExample([...sequence, letter, number]);
          }
        }
      }
    } else {
      for (let i = 0; i < availableLetters.length; i++) {
        const letter = availableLetters[i];
        const newAvailableLetters = removeElementAt(availableLetters, i);
        for (let j = 0; j < availableNumbers.length; j++) {
          const number = availableNumbers[j];
          const newAvailableNumbers = removeElementAt(availableNumbers, j);
          const newMapping = insert(mapping, letter, number);
          generate(n - 1, [...sequence, letter, number], newMapping, newAvailableLetters, newAvailableNumbers);
        }
      }
    }
  }

  generate(2, [], new Map(), allLetters, allNumbers);

  // Convert to tensors with embeddings
  const numExamples = inputArray.length;
  const embeddedInputArray = inputArray.map(embedInput);

  // Dispose old tensors if they exist
  if (trainingInputTensor) {
    try { trainingInputTensor.dispose(); } catch (e) { /* ignore */ }
  }
  if (trainingOutputTensor) {
    try { trainingOutputTensor.dispose(); } catch (e) { /* ignore */ }
  }

  trainingInputArray = inputArray;
  trainingOutputArray = outputArray;
  trainingInputTensor = tf.tensor2d(embeddedInputArray, [numExamples, EMBEDDED_INPUT_SIZE]);
  trainingOutputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;
}

// Getters for training data
function getTrainingInputArray(): number[][] {
  return trainingInputArray;
}

function getTrainingOutputArray(): number[] {
  return trainingOutputArray;
}

function getTrainingInputTensor(): Tensor2D | null {
  return trainingInputTensor;
}

function getTrainingOutputTensor(): Tensor2D | null {
  return trainingOutputTensor;
}

// Dispose training data tensors
function disposeTrainingData() {
  if (trainingInputTensor) {
    try { trainingInputTensor.dispose(); } catch (e) { /* ignore */ }
    trainingInputTensor = null;
  }
  if (trainingOutputTensor) {
    try { trainingOutputTensor.dispose(); } catch (e) { /* ignore */ }
    trainingOutputTensor = null;
  }
}

export {
  generateData,
  getTrainingInputArray,
  getTrainingOutputArray,
  getTrainingInputTensor,
  getTrainingOutputTensor,
  disposeTrainingData
};

