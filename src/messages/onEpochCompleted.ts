/**
 * Message handler for epoch completion.
 * 
 * This message is sent when a training epoch completes.
 */

import { Schedule } from "../messageLoop.js";

// Message type for epoch completion
export type OnEpochCompletedMsg = {
  type: "OnEpochCompleted";
  epoch: number;
  loss: number;
};

// Type for the message handler (used by module implementations)
export type OnEpochCompletedHandler = (schedule: Schedule, epoch: number, loss: number) => void;
