import { AppState, DomElements } from "./types.js";
import { trainingStep } from "./model.js";
import { ReinitializeModelImpl } from "./orchestrators/reinitializeModel.js";

async function toggleTrainingMode(appState: AppState, dom: DomElements) {
  appState.isTraining = !appState.isTraining;
  const trainButton = dom.trainButton;

  if (appState.isTraining) {
    trainButton.innerText = 'Pause';
    requestAnimationFrame(() => trainingStep(appState, dom));
  } else {
    trainButton.innerText = 'Train Model';
  }
}

// Implementation for the reinitializeModel orchestrator
// Currently a no-op, but can be extended in the future
const reinitializeModel: ReinitializeModelImpl = (appState, dom) => {
  // No-op for now - ui-controls doesn't need to do anything on model reinitialize
};

export { toggleTrainingMode, reinitializeModel };
