/**
 * Orchestrators for training mode control.
 * 
 * Contains orchestrators for starting and stopping training.
 */

// Type for the startTraining orchestrator function
export type StartTraining = () => void;

// Type for the stopTraining orchestrator function
export type StopTraining = () => void;

// Extend the Window interface to include our orchestrators
declare global {
  interface Window {
    startTraining: StartTraining;
    stopTraining: StopTraining;
  }
}
