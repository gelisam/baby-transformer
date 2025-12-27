/**
 * Orchestrator for toggling training mode.
 * 
 * This orchestrator is triggered when the user starts or stops training.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type ToggleTraining = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    toggleTraining: ToggleTraining;
  }
}
