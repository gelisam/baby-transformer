/**
 * Message handler for setting model weights.
 * 
 * Used by perfect-weights.ts to push computed weights to the model.
 */

import { Schedule } from "../messageLoop.js";
import { Tensor } from "../tf.js";

export type SetModelWeightsMsg = {
  type: "SetModelWeights";
  weights: Tensor[];
};

export type SetModelWeightsHandler = (schedule: Schedule, weights: Tensor[]) => void;
