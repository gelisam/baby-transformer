import { Tensor2D } from "./tf.js";

// TrainingData is kept for internal use within modules if needed
// but is no longer passed around between modules
interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

// Note: AppState and DomElements have been removed in favor of
// module-local state. Each module now manages its own globals
// and communicates through orchestrator functions.

export { TrainingData };
