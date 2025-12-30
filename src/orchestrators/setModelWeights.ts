/**
 * Orchestrator for setting model weights.
 * 
 * Used by perfect-weights.ts to push computed weights to the model.
 */

import { Tensor } from "../tf.js";

// Type for the setModelWeights orchestrator function
export type SetModelWeights = (weights: Tensor[]) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    setModelWeights: SetModelWeights;
  }
}
