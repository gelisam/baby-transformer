/**
 * Orchestrator for stopping training mode.
 * 
 * This orchestrator is triggered when training is paused/stopped.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type StopTraining = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    stopTraining: StopTraining;
  }
}
