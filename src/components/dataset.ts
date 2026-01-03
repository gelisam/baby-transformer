import {
  INPUT_SIZE,
  OUTPUT_SIZE,
} from "../constants.js";
import type { InputFormat } from "../constants.js";
import { tf, Tensor2D } from "../tf.js";
import {
  NUMBERS,
  LETTERS,
  tokenNumberToIndex,
  tokenStringToTokenNumber
} from "../tokens.js";
import { TrainingData, SetTrainingDataMsg } from "../messages/setTrainingData.js";

// Pure function: Generate training data for the classification task
function generateData(inputFormat: InputFormat): TrainingData {
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
      for (const letter of mapping.keys()) {
        addExample([...sequence, letter, mapping.get(letter)!]);
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

  // Convert to tensors
  // inputArray contains token numbers (1-6), but inputTensor needs token indices (0-5)
  // for the embedding layer to work correctly
  const numExamples = inputArray.length;
  const inputIndicesArray = inputArray.map(input => input.map(tokenNumberToIndex));

  const inputTensor = tf.tensor2d(inputIndicesArray, [numExamples, INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  return { inputArray, outputArray, inputTensor, outputTensor };
}

import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";

// Implementation of the reinitializeModel message handler
// Generates new training data and pushes to other modules via setTrainingData message
const reinitializeModel: ReinitializeModelHandler = (schedule, _numLayers, _neuronsPerLayer, inputFormat, _outputFormat) => {
  const data = generateData(inputFormat);
  schedule({ type: "SetTrainingData", data } as SetTrainingDataMsg);
};

export {
  generateData,
  reinitializeModel
};
