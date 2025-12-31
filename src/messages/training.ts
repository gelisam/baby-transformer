/**
 * Message handlers for training mode control.
 * 
 * Contains messages for starting and stopping training.
 */

import { Schedule } from "../messageLoop.js";

// Message type for start training
export type StartTrainingMsg = {
  type: "StartTraining";
};

// Message type for stop training
export type StopTrainingMsg = {
  type: "StopTraining";
};

// Type for the startTraining message handler
export type StartTrainingHandler = (schedule: Schedule) => void;

// Type for the stopTraining message handler
export type StopTrainingHandler = (schedule: Schedule) => void;
