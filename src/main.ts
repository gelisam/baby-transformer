import { setBackend } from "./tf.js";
import { Msg, Schedule } from "./messageLoop.js";
import type { InputFormat } from "./constants.js";
import { OnEpochCompletedMsg } from "./messages/onEpochCompleted.js";
import { RefreshVizMsg } from "./messages/refreshViz.js";
import { ReinitializeModelMsg } from "./messages/reinitializeModel.js";
import { SetModelWeightsMsg } from "./messages/setModelWeights.js";
import { SetTrainingDataMsg } from "./messages/setTrainingData.js";
import { StartTrainingMsg, StopTrainingMsg } from "./messages/training.js";
import * as dataset from "./components/dataset.js";
import * as model from "./components/model.js";
import * as perfectWeights from "./components/perfect-weights.js";
import * as uiControls from "./components/ui-controls.js";
import * as viz from "./components/viz.js";

// Module-local state for layer configuration
let numLayers = 4;
let neuronsPerLayer = 6;
let inputFormat: InputFormat = 'embedding';

// Process a single message
function processMessage(schedule: Schedule, msg: Msg): void {
  switch (msg.type) {
    case "ReinitializeModel": {
      const m = msg as ReinitializeModelMsg;
      // 1. First, create a new model (model.ts)
      model.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 2. Generate new data and push to modules via message (dataset.ts)
      dataset.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 3. Update visualization (viz.ts)
      viz.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 4. Update perfect weights button state (perfect-weights.ts)
      perfectWeights.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 5. Update UI controls state (ui-controls.ts)
      uiControls.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 6. Set ready status
      viz.setStatusMessage('Ready to train!');
      break;
    }
    case "RefreshViz": {
      viz.refreshViz(schedule);
      break;
    }
    case "OnEpochCompleted": {
      const m = msg as OnEpochCompletedMsg;
      viz.onEpochCompleted(schedule, m.epoch, m.loss);
      break;
    }
    case "StartTraining": {
      // 1. Start training in model.ts (start training loop)
      model.startTraining(schedule);
      
      // 2. Update UI in ui-controls.ts (button text)
      uiControls.startTraining(schedule);
      break;
    }
    case "StopTraining": {
      // 1. Stop training in model.ts (stop training loop)
      model.stopTraining(schedule);
      
      // 2. Update UI in ui-controls.ts (button text)
      uiControls.stopTraining(schedule);
      break;
    }
    case "SetTrainingData": {
      const m = msg as SetTrainingDataMsg;
      // 1. Set training data in model.ts
      model.setTrainingData(schedule, m.data);
      
      // 2. Set training data reference in viz.ts for lookup
      viz.setTrainingData(schedule, m.data);
      break;
    }
    case "SetModelWeights": {
      const m = msg as SetModelWeightsMsg;
      // Set model weights in model.ts
      model.setModelWeights(schedule, m.weights);
      break;
    }
    default:
      console.warn(`Unknown message type: ${msg.type}`);
  }
}

// Define the global message loop
window.messageLoop = (msgOrMsgs: Msg | Msg[]): void => {
  // Create a distinct message queue for this invocation
  const messageQueue: Msg[] = [];
  
  // Create a schedule function that adds messages to this invocation's queue
  const schedule: Schedule = (msg: Msg) => {
    messageQueue.push(msg);
  };
  
  // Add initial message(s) to the queue
  if (Array.isArray(msgOrMsgs)) {
    messageQueue.push(...msgOrMsgs);
  } else {
    messageQueue.push(msgOrMsgs);
  }
  
  // Process all messages in the queue
  while (messageQueue.length > 0) {
    const msg = messageQueue.shift()!;
    processMessage(schedule, msg);
  }
};

// Helper function to stop training and dispose tensors before reinitializing
function prepareForReinitialize(): void {
  // Always stop training (safe to call even if not training)
  window.messageLoop({ type: "StopTraining" } as StopTrainingMsg);
  model.disposeTrainingData();
  viz.disposeVizData();
}

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize module DOM references (each module calls document.getElementById directly)
  viz.initVizDom();
  uiControls.initUiControlsDom();
  perfectWeights.initPerfectWeightsDom();

  // Get DOM elements needed only by main.ts for event listeners
  const backendSelector = document.getElementById('backend-selector') as HTMLSelectElement;
  const numLayersSlider = document.getElementById('num-layers-slider') as HTMLInputElement;
  const numLayersValue = document.getElementById('num-layers-value') as HTMLSpanElement;
  const neuronsPerLayerSlider = document.getElementById('neurons-per-layer-slider') as HTMLInputElement;
  const neuronsPerLayerValue = document.getElementById('neurons-per-layer-value') as HTMLSpanElement;
  const inputFormatSelector = document.getElementById('input-format-selector') as HTMLSelectElement;

  // Add event listener for backend changes
  backendSelector.addEventListener('change', async () => {
    prepareForReinitialize();
    await setBackend(backendSelector.value);
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });

  // Add event listeners for layer configuration sliders
  numLayersSlider.addEventListener('input', () => {
    numLayers = parseInt(numLayersSlider.value, 10);
    numLayersValue.textContent = numLayers.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });

  neuronsPerLayerSlider.addEventListener('input', () => {
    neuronsPerLayer = parseInt(neuronsPerLayerSlider.value, 10);
    neuronsPerLayerValue.textContent = neuronsPerLayer.toString();
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });

  // Add event listener for input format changes
  inputFormatSelector.addEventListener('change', () => {
    inputFormat = inputFormatSelector.value as InputFormat;
    prepareForReinitialize();
    window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
  });

  // Set up input textbox event listeners in viz module
  viz.setupInputEventListeners();

  // Initial setup
  await setBackend(backendSelector.value);
  window.messageLoop({ type: "ReinitializeModel", numLayers, neuronsPerLayer, inputFormat } as ReinitializeModelMsg);
});
