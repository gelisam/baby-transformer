/**
 * Message handler for model re-initialization.
 * 
 * This message is sent when the model needs to be re-initialized,
 * such as when:
 * - The backend changes
 * - Layer configuration changes (number of layers, neurons per layer)
 * - Input format changes
 * 
 * Each module can define its own implementation of `reinitializeModel` that
 * will be called by the main implementation in the appropriate order.
 * 
 * Arguments are passed explicitly rather than through shared state objects.
 */

import { Schedule } from "../messageLoop.js";
import type { InputFormat } from "../constants.js";

export type ReinitializeModelMsg = {
  type: "ReinitializeModel";
  numLayers: number;
  neuronsPerLayer: number;
  inputFormat: InputFormat;
};

export type ReinitializeModelHandler = (schedule: Schedule, numLayers: number, neuronsPerLayer: number, inputFormat: InputFormat) => void;
