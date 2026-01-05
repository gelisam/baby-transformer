/**
 * Message handler for epoch completion.
 * 
 * This message is sent when a training epoch completes.
 */

import { Schedule } from "../messageLoop.js";

export type OnEpochCompletedMsg = {
  type: "OnEpochCompleted";
  epoch: number;
  loss: number;
};

export type OnEpochCompletedHandler = (schedule: Schedule, epoch: number, loss: number) => void;
