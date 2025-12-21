/**
 * Orchestrator for toggling training mode.
 * 
 * This orchestrator is triggered when the user starts or stops training.
 */

// Type definition for module-specific implementations
export type ToggleTrainingImpl = () => void;

// Type for the global orchestrator function
export type ToggleTrainingOrchestrator = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    toggleTraining: ToggleTrainingOrchestrator;
  }
}
