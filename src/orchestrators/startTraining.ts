/**
 * Orchestrator for starting training mode.
 * 
 * This orchestrator is triggered when training is started.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type StartTraining = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    startTraining: StartTraining;
  }
}
