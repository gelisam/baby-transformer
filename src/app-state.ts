import { Sequential } from "./tf.js";
import { TrainingData, AppState } from "./types.js";

const appState: AppState = {
  model: undefined as unknown as Sequential,
  isTraining: false,
  currentEpoch: 0,
  lossHistory: [] as { epoch: number, loss: number }[],
  data: undefined as unknown as TrainingData,
  vizData: undefined as unknown as TrainingData,
  num_layers: 4,
  neurons_per_layer: 6
};

export { appState };
