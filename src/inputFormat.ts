import { EMBEDDING_DIM } from "./embeddings.js";

const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____

// Input format type - determines how inputs are transformed before the first layer
type InputFormat = 'number' | 'one-hot' | 'embedding';

// Calculate the size after preprocessing based on input format
function getTransformedInputSize(inputFormat: InputFormat, vocabSize: number): number {
  if (inputFormat === 'embedding') {
    return INPUT_SIZE * EMBEDDING_DIM;
  } else if (inputFormat === 'one-hot') {
    return INPUT_SIZE * vocabSize * 2; // vocabSize * 2 tokens total
  } else {
    return INPUT_SIZE;
  }
}

export {
  INPUT_SIZE,
  getTransformedInputSize,
};
export type { InputFormat };
