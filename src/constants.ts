import { TOKENS } from "./tokens.js";

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const OUTPUT_SIZE = TOKENS.length; // a probability for each possible output token

export {
  INPUT_SIZE,
  OUTPUT_SIZE,
};
