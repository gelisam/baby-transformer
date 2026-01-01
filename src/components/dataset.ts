import {
  INPUT_SIZE,
  OUTPUT_SIZE,
} from "../constants.js";
import type { InputFormat } from "../constants.js";
import { EMBEDDED_INPUT_SIZE, embedInput } from "../embeddings.js";
import { tf, Tensor2D } from "../tf.js";
import {
  NUMBERS,
  LETTERS,
  TOKENS,
  tokenNumberToIndex,
  tokenStringToTokenNumber
} from "../tokens.js";
import { Schedule } from "../messageLoop.js";
import { TrainingData, SetTrainingDataMsg } from "../messages/setTrainingData.js";
import { getInputSizeForFormat } from "./model.js";

// Transform a single token to one-hot representation
function tokenToOneHot(tokenNum: number): number[] {
  const index = tokenNumberToIndex(tokenNum);
  const oneHot = Array(TOKENS.length).fill(0);
  oneHot[index] = 1;
  return oneHot;
}

// Transform an input array based on the input format
function transformInput(input: number[], inputFormat: InputFormat): number[] {
  switch (inputFormat) {
    case 'number':
      // Return raw token numbers as-is (values 1-6)
      return input;
    case 'one-hot':
      // Return concatenated one-hot vectors
      return input.flatMap(tokenToOneHot);
    case 'embedding':
      // Return concatenated embeddings
      return embedInput(input);
  }
}

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

  // Convert to tensors based on input format
  const numExamples = inputArray.length;
  const transformedInputArray = inputArray.map(input => transformInput(input, inputFormat));
  const inputSize = getInputSizeForFormat(inputFormat);

  const inputTensor = tf.tensor2d(transformedInputArray, [numExamples, inputSize]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  return { inputArray, outputArray, inputTensor, outputTensor };
}

import { ReinitializeModelHandler } from "../messages/reinitializeModel.js";

// Implementation of the reinitializeModel message handler
// Generates new training data and pushes to other modules via setTrainingData message
const reinitializeModel: ReinitializeModelHandler = (schedule, _numLayers, _neuronsPerLayer, inputFormat) => {
  const data = generateData(inputFormat);
  schedule({ type: "SetTrainingData", data } as SetTrainingDataMsg);
};

export {
  generateData,
  reinitializeModel,
  transformInput
};
