import { Msg, Schedule } from "../messageLoop.js";

export interface InitMsg extends Msg {
  type: "Init";
}

export type InitHandler = (schedule: Schedule) => void;
