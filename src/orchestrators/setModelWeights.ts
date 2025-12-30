/**
 * Orchestrator for setting model weights.
 * 
 * Used by perfect-weights.ts to push computed weights to the model.
 */

import { tf } from "../tf.js";

// Type alias for TensorFlow Tensor
type Tensor = ReturnType<typeof tf.tensor>;

// Type for the setModelWeights orchestrator function
export type SetModelWeights = (weights: Tensor[]) => void;

// Extend the Window interface to include our orchestrator
declare global {
  interface Window {
    setModelWeights: SetModelWeights;
  }
}
