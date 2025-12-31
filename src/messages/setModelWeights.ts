/**
 * Message handler for setting model weights.
 * 
 * Used by perfect-weights.ts to push computed weights to the model.
 */

import { Schedule } from "../messageLoop.js";
import { Tensor } from "../tf.js";

// Message type for set model weights
export type SetModelWeightsMsg = {
  type: "SetModelWeights";
  weights: Tensor[];
};

// Type for the message handler (used by module implementations)
export type SetModelWeightsHandler = (schedule: Schedule, weights: Tensor[]) => void;
