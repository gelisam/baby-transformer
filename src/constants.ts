const EXAMPLES_GIVEN = 2;
const INPUT_SIZE = EXAMPLES_GIVEN * 2 + 1;  // <letter>=<number> <letter>=<number> <letter>=____

const EPOCHS_PER_BATCH = 1;

const EMBEDDING_DIM = 3;

// Input format type - determines how inputs are transformed before the first layer
type InputFormat = 'number' | 'one-hot' | 'embedding';

// Calculate the output size based on vocabulary size
function getOutputSize(vocabSize: number): number {
  return vocabSize * 2; // vocabSize numbers + vocabSize letters
}

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
  EPOCHS_PER_BATCH,
  EMBEDDING_DIM,
  getOutputSize,
  getTransformedInputSize,
};
export type { InputFormat };
