/**
 * Message handlers for training mode control.
 * 
 * Contains messages for starting and stopping training.
 */

import { Schedule } from "../messageLoop.js";

export type StartTrainingMsg = {
  type: "StartTraining";
};

export type StopTrainingMsg = {
  type: "StopTraining";
};

export type StartTrainingHandler = (schedule: Schedule) => void;

export type StopTrainingHandler = (schedule: Schedule) => void;
