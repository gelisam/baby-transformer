import { TOKENS } from "./tokens.js";

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const OUTPUT_SIZE = TOKENS.length; // a probability for each possible output token

const EPOCHS_PER_BATCH = 1;

// Input format type - determines how inputs are transformed before the first layer
type InputFormat = 'number' | 'one-hot' | 'embedding';

export {
  INPUT_SIZE,
  OUTPUT_SIZE,
  EPOCHS_PER_BATCH,
};
export type { InputFormat };
