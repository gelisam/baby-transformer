/**
 * Orchestrator for model re-initialization.
 * 
 * This orchestrator is triggered when the model needs to be re-initialized,
 * such as when:
 * - The backend changes
 * - Layer configuration changes (number of layers, neurons per layer)
 * 
 * Each module can define its own implementation of `reinitializeModel` that
 * will be called by the main implementation in the appropriate order.
 * 
 * Arguments are passed explicitly rather than through shared state objects.
 */

// Type for the orchestrator function (used by both window and module implementations)
export type ReinitializeModel = (numLayers: number, neuronsPerLayer: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    reinitializeModel: ReinitializeModel;
  }
}
