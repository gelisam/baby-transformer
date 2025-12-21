/**
 * Orchestrator for visualization refresh.
 * 
 * This orchestrator is triggered when the visualization needs to be updated,
 * such as during training or when input data changes.
 */

// Type definition for module-specific implementations
export type RefreshVizImpl = () => void;

// Type for the global orchestrator function
export type RefreshVizOrchestrator = () => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    refreshViz: RefreshVizOrchestrator;
  }
}
