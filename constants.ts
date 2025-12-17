import { TOKENS } from "./tokens.js";

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const OUTPUT_SIZE = TOKENS.length; // a probability for each possible output token

const EPOCHS_PER_BATCH = 1;

const EMBEDDING_DIM = 2;
const EMBEDDED_INPUT_SIZE = INPUT_SIZE * EMBEDDING_DIM;

export {
  INPUT_SIZE,
  OUTPUT_SIZE,
  EPOCHS_PER_BATCH,
  EMBEDDING_DIM,
  EMBEDDED_INPUT_SIZE
};
