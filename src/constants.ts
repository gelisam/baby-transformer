import { TOKENS } from "./tokens.js";

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____
const OUTPUT_SIZE = TOKENS.length; // a probability for each possible output token

const EPOCHS_PER_BATCH = 1;

const EMBEDDING_DIM = 3;

// Input format type - determines how inputs are transformed before the first layer
type InputFormat = 'number' | 'one-hot' | 'embedding';

// Calculate the size after preprocessing based on input format
function getTransformedInputSize(inputFormat: InputFormat): number {
  const vocabSize = TOKENS.length;
  if (inputFormat === 'embedding') {
    return INPUT_SIZE * EMBEDDING_DIM;
  } else if (inputFormat === 'one-hot') {
    return INPUT_SIZE * vocabSize;
  } else {
    return INPUT_SIZE;
  }
}

export {
  INPUT_SIZE,
  OUTPUT_SIZE,
  EPOCHS_PER_BATCH,
  EMBEDDING_DIM,
  getTransformedInputSize,
};
export type { InputFormat };
