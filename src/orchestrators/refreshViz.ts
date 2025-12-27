/**
 * Orchestrator for visualization refresh.
 * 
 * This orchestrator is triggered when the visualization needs to be updated,
 * such as during training or when input data changes.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type RefreshViz = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    refreshViz: RefreshViz;
  }
}
