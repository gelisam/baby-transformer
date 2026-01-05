/**
 * Message handler for setting training data.
 * 
 * This message is sent when new training data is generated,
 * passing the data to all modules that need it.
 */

import { Schedule } from "../messageLoop.js";
import { Tensor2D } from "../tf.js";

export interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

export type SetTrainingDataMsg = {
  type: "SetTrainingData";
  data: TrainingData;
};

export type SetTrainingDataHandler = (schedule: Schedule, data: TrainingData) => void;
