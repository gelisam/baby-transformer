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

// Type definition for module-specific implementations
export type ReinitializeModelImpl = (numLayers: number, neuronsPerLayer: number) => void;

// Type for the global orchestrator function
export type ReinitializeModelOrchestrator = (numLayers: number, neuronsPerLayer: number) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    reinitializeModel: ReinitializeModelOrchestrator;
  }
}
