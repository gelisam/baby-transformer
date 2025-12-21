/**
 * Orchestrator for training status updates.
 * 
 * This orchestrator is triggered when training status changes,
 * such as during each training epoch.
 */

// Type definition for module-specific implementations
export type UpdateTrainingStatusImpl = (epoch: number, loss: number) => void;

// Type for the global orchestrator function
export type UpdateTrainingStatusOrchestrator = (epoch: number, loss: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    updateTrainingStatus: UpdateTrainingStatusOrchestrator;
  }
}
