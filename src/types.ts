import { Tensor2D, Sequential } from "./tf.js";

interface TrainingData {
  inputArray: number[][];
  outputArray: number[];
  inputTensor: Tensor2D;
  outputTensor: Tensor2D;
}

interface AppState {
  model: Sequential;
  isTraining: boolean;
  currentEpoch: number;
  lossHistory: { epoch: number; loss: number }[];
  data: TrainingData;
  vizData: TrainingData;
  num_layers: number;
  neurons_per_layer: number;
}

interface DomElements {
  trainButton: HTMLButtonElement;
  perfectWeightsButton: HTMLButtonElement;
  perfectWeightsTooltipText: HTMLSpanElement;
  backendSelector: HTMLSelectElement;
  numLayersSlider: HTMLInputElement;
  numLayersValue: HTMLSpanElement;
  neuronsPerLayerSlider: HTMLInputElement;
  neuronsPerLayerValue: HTMLSpanElement;
  inputElements: HTMLInputElement[];
  statusElement: HTMLElement;
  outputCanvas: HTMLCanvasElement;
  lossCanvas: HTMLCanvasElement;
  networkCanvas: HTMLCanvasElement;
  toaster: HTMLElement | null;
}

export { TrainingData, AppState, DomElements };
