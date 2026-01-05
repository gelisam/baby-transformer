/**
 * Message handler for visualization refresh.
 * 
 * This message is sent when the visualization needs to be updated,
 * such as during training or when input data changes.
 */

import { Schedule } from "../messageLoop.js";

export type RefreshVizMsg = {
  type: "RefreshViz";
};

export type RefreshVizHandler = (schedule: Schedule) => void;
