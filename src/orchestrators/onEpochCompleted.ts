/**
 * Orchestrator for epoch completion.
 * 
 * This orchestrator is triggered when a training epoch completes.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type OnEpochCompleted = (epoch: number, loss: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    onEpochCompleted: OnEpochCompleted;
  }
}
