/**
 * Message handler for visualization refresh.
 * 
 * This message is sent when the visualization needs to be updated,
 * such as during training or when input data changes.
 */

import { Schedule } from "../messageLoop.js";

// Message type for refresh viz
export type RefreshVizMsg = {
  type: "RefreshViz";
};

// Type for the message handler (used by module implementations)
export type RefreshVizHandler = (schedule: Schedule) => void;
