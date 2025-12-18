/**
 * Data management utilities for creating and manipulating training data.
 * 
 * This module handles the conversion between raw data arrays and TensorFlow tensors,
 * including proper embedding and one-hot encoding.
 */

import { OUTPUT_SIZE } from "./constants.js";
import { EMBEDDED_INPUT_SIZE, embedInput } from "./embeddings.js";
import { tf, Tensor2D } from "./tf.js";
import { TrainingData } from "./types.js";
import { tokenNumberToIndex } from "./tokens.js";

/**
 * Create training data from input and output arrays.
 * 
 * Converts raw arrays into embedded tensors suitable for model training.
 * 
 * @param inputArray - Array of input sequences (token numbers)
 * @param outputArray - Array of expected output tokens
 * @returns TrainingData object containing both arrays and tensors
 */
function createTrainingData(
  inputArray: number[][],
  outputArray: number[]
): TrainingData {
  const numExamples = inputArray.length;
  const embeddedInputArray = inputArray.map(embedInput);

  const inputTensor = tf.tensor2d(embeddedInputArray, [numExamples, EMBEDDED_INPUT_SIZE]);
  const outputTensor = tf.oneHot(outputArray.map(tokenNumberToIndex), OUTPUT_SIZE) as Tensor2D;

  return {
    inputArray,
    outputArray,
    inputTensor,
    outputTensor
  };
}

/**
 * Pick random examples from training data.
 * 
 * @param data - The source training data
 * @param count - Number of random examples to pick
 * @returns Object containing input and output arrays for the selected examples
 */
function pickRandomExamples(
  data: TrainingData,
  count: number
): { inputArray: number[][], outputArray: number[] } {
  const inputArray: number[][] = [];
  const outputArray: number[] = [];

  for (let i = 0; i < count; i++) {
    const randomIndex = Math.floor(Math.random() * data.inputArray.length);
    inputArray.push(data.inputArray[randomIndex]);
    outputArray.push(data.outputArray[randomIndex]);
  }

  return { inputArray, outputArray };
}

export { createTrainingData, pickRandomExamples };
