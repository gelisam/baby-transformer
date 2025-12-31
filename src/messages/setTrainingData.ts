/**
 * Message handler for setting training data.
 * 
 * This message is sent when new training data is generated,
 * passing the data to all modules that need it.
 */

import { Schedule } from "../messageLoop.js";
import { Tensor2D } from "../tf.js";

// Training data structure
export interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

// Message type for set training data
export type SetTrainingDataMsg = {
  type: "SetTrainingData";
  data: TrainingData;
};

// Type for the message handler (used by module implementations)
export type SetTrainingDataHandler = (schedule: Schedule, data: TrainingData) => void;
