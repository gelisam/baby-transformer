import { transposeArray } from "./array.js";
import { INPUT_SIZE } from "./constants.js";
import { tokenNumberToIndex } from "./tokens.js";

const EMBEDDING_DIM = 2;
const EMBEDDED_INPUT_SIZE = INPUT_SIZE * EMBEDDING_DIM;

const EMBEDDING_MATRIX: number[][] = [
  [0, 1],  // "1 "
  [0, 2],  // "2 "
  [0, 3],  // "3 "
  [1, 0],  // "A="
  [2, 0],  // "B="
  [3, 0]   // "C="
];
const UNEMBEDDING_MATRIX = transposeArray(EMBEDDING_MATRIX);

function embedTokenNumber(tokenNum: number): number[] {
  const tokenIndex = tokenNumberToIndex(tokenNum);
  return EMBEDDING_MATRIX[tokenIndex];
}

function embedInput(input: number[]): number[] {
  const embeddedInput: number[] = [];
  for (let i = 0; i < input.length; i++) {
    const embedding = embedTokenNumber(input[i]);
    embeddedInput.push(embedding[0], embedding[1]);
  }
  return embeddedInput;
}

export {
  EMBEDDING_DIM,
  EMBEDDED_INPUT_SIZE,
  EMBEDDING_MATRIX,
  UNEMBEDDING_MATRIX,
  embedTokenNumber,
  embedInput
};
