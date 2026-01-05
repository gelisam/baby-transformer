/**
 * Message Loop Pattern
 * 
 * This module defines the core types for the message loop pattern,
 * which replaces the orchestrator pattern for cross-component communication.
 */

export type Msg = { type: string };

export type Schedule = (msg: Msg) => void;

export type MessageLoop = (msg: Msg | Msg[]) => void;

declare global {
  interface Window {
    messageLoop: MessageLoop;
  }
}
