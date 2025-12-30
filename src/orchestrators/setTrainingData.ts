/**
 * Orchestrator for setting training data.
 * 
 * This orchestrator is triggered when new training data is generated,
 * passing the data to all modules that need it.
 */

import { Tensor2D } from "../tf.js";

// Training data structure
export interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

// Type for the orchestrator function (used by both window and module implementations)
export type SetTrainingData = (data: TrainingData) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    setTrainingData: SetTrainingData;
  }
}
