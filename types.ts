import { Tensor2D } from "./tf.js";

interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

export { TrainingData };
