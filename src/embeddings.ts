import { transposeArray } from "./array.js";
import { INPUT_SIZE } from "./constants.js";
import { tokenNumberToIndex } from "./tokens.js";

const EMBEDDING_DIM = 3;
const EMBEDDED_INPUT_SIZE = INPUT_SIZE * EMBEDDING_DIM;

const SQRT3_OVER_2 = Math.sqrt(3) / 2;

const EMBEDDING_MATRIX: number[][] = [
  [10, 1, 0],                // "1 "
  [10, -0.5, SQRT3_OVER_2],  // "2 "
  [0, -0.5, -SQRT3_OVER_2],  // "3 "
  [1, 10, 0],                // "A="
  [-0.5, 10, SQRT3_OVER_2],  // "B="
  [-0.5, 10, -SQRT3_OVER_2]  // "C="
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
    embeddedInput.push(...embedding);
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
