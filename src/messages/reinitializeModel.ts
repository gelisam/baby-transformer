/**
 * Message handler for model re-initialization.
 * 
 * This message is sent when the model needs to be re-initialized,
 * such as when:
 * - The backend changes
 * - Layer configuration changes (number of layers, neurons per layer)
 * 
 * Each module can define its own implementation of `reinitializeModel` that
 * will be called by the main implementation in the appropriate order.
 * 
 * Arguments are passed explicitly rather than through shared state objects.
 */

import { Schedule } from "../messageLoop.js";

// Message type for reinitialize model
export type ReinitializeModelMsg = {
  type: "ReinitializeModel";
  numLayers: number;
  neuronsPerLayer: number;
};

// Type for the message handler (used by module implementations)
export type ReinitializeModelHandler = (schedule: Schedule, numLayers: number, neuronsPerLayer: number) => void;
