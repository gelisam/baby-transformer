/**
 * Orchestrator for training step completion.
 * 
 * This orchestrator is triggered when a training step completes,
 * such as after each training epoch.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type OnTrainingStepCompleted = (epoch: number, loss: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    onTrainingStepCompleted: OnTrainingStepCompleted;
  }
}
