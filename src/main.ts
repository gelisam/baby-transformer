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

function processMessage(schedule: Schedule, msg: Msg): void {
  switch (msg.type) {
    case "Init": {
      vizExamples.init(schedule);
      uiControls.init(schedule);
      perfectWeights.init(schedule);
      modelConfig.init(schedule);
      break;
    }
    case "ReinitializeModel": {
      const m = msg as ReinitializeModelMsg;
      model.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      dataset.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      vizArchitecture.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      vizLoss.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      perfectWeights.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
      uiControls.reinitializeModel(schedule, m.numLayers, m.neuronsPerLayer, m.inputFormat);
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
      model.startTraining(schedule);
      uiControls.startTraining(schedule);
      break;
    }
    case "StopTraining": {
      model.stopTraining(schedule);
      uiControls.stopTraining(schedule);
      break;
    }
    case "SetTrainingData": {
      const m = msg as SetTrainingDataMsg;
      model.setTrainingData(schedule, m.data);
      vizExamples.setTrainingData(schedule, m.data);
      break;
    }
    case "SetModelWeights": {
      const m = msg as SetModelWeightsMsg;
      model.setModelWeights(schedule, m.weights);
      break;
    }
    default:
      console.warn(`Unknown message type: ${msg.type}`);
  }
}

window.messageLoop = (msgOrMsgs: Msg | Msg[]): void => {
  const messageQueue: Msg[] = [];
  
  const schedule: Schedule = (msg: Msg) => {
    messageQueue.push(msg);
  };
  
  if (Array.isArray(msgOrMsgs)) {
    messageQueue.push(...msgOrMsgs);
  } else {
    messageQueue.push(msgOrMsgs);
  }
  
  while (messageQueue.length > 0) {
    const msg = messageQueue.shift()!;
    processMessage(schedule, msg);
  }
};

document.addEventListener('DOMContentLoaded', async () => {
  window.messageLoop({ type: "Init" } as InitMsg);
  await modelConfig.performInitialSetup();
});
