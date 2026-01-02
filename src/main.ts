import { Msg, Schedule } from "./messageLoop.js";
import { InitMsg } from "./messages/init.js";
import { OnEpochCompletedMsg } from "./messages/onEpochCompleted.js";
import { RefreshVizMsg } from "./messages/refreshViz.js";
import { ReinitializeModelMsg } from "./messages/reinitializeModel.js";
import { SetModelWeightsMsg } from "./messages/setModelWeights.js";
import { SetTrainingDataMsg } from "./messages/setTrainingData.js";
import { StartTrainingMsg, StopTrainingMsg } from "./messages/training.js";
import * as dataset from "./components/dataset.js";
import * as model from "./components/model.js";
import * as modelConfig from "./components/model-config.js";
import * as perfectWeights from "./components/perfect-weights.js";
import * as uiControls from "./components/ui-controls.js";
import * as vizLoss from "./components/viz-loss.js";
import * as vizArchitecture from "./components/viz-architecture.js";
import * as vizExamples from "./components/viz-examples.js";

// Process a single message
function processMessage(schedule: Schedule, msg: Msg): void {
  switch (msg.type) {
    case "Init": {
      // Initialize all components that need event listeners attached
      vizLoss.init(schedule);
      vizArchitecture.init(schedule);
      vizExamples.init(schedule);
      uiControls.init(schedule);
      perfectWeights.init(schedule);
      modelConfig.init(schedule);
      break;
    }
    case "ReinitializeModel": {
      const m = msg as ReinitializeModelMsg;
      // 1. First, create a new model (model.ts)
      model.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 2. Generate new data and push to modules via message (dataset.ts)
      dataset.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 3. Update visualizations (split into three components)
      vizExamples.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      vizArchitecture.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 4. Update perfect weights button state (perfect-weights.ts)
      perfectWeights.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 5. Update UI controls state (ui-controls.ts)
      uiControls.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);

      // 6. Set ready status
      vizLoss.setStatusMessage('Ready to train!');
      break;
    }
    case "RefreshViz": {
      vizExamples.refreshViz(schedule);
      vizLoss.refreshViz(schedule);
      break;
    }
    case "OnEpochCompleted": {
      const m = msg as OnEpochCompletedMsg;
      vizLoss.onEpochCompleted(schedule, m.epoch, m.loss);
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
      
      // 2. Set training data reference in viz-examples.ts for lookup
      vizExamples.setTrainingData(schedule, m.data);
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

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize all components by sending the Init message
  window.messageLoop({ type: "Init" } as InitMsg);

  // Initial setup
  await modelConfig.performInitialSetup();
});
