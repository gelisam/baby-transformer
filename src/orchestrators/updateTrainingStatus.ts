/**
 * Orchestrator for training status updates.
 * 
 * This orchestrator is triggered when training status changes,
 * such as during each training epoch.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type UpdateTrainingStatus = (epoch: number, loss: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    updateTrainingStatus: UpdateTrainingStatus;
  }
}
