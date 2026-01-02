import { Msg, Schedule } from "../messageLoop.js";

// Init message - sent once when the application starts to attach event listeners
export interface InitMsg extends Msg {
  type: "Init";
}

// Handler type for components that need to attach event listeners during initialization
export type InitHandler = (schedule: Schedule) => void;
