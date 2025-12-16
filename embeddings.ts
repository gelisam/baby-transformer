import { EMBEDDING_DIM } from "./constants.js";
import { tokenNumberToIndex } from "./tokens.js";

const EMBEDDING_MATRIX: number[][] = [
  [0, 1],  // "1 "
  [0, 2],  // "2 "
  [0, 3],  // "3 "
  [1, 0],  // "A="
  [2, 0],  // "B="
  [3, 0]   // "C="
];
function transposeArray(matrix: number[][]): number[][] {
  const transposed: number[][] = [];
  for (let i = 0; i < matrix[0].length; i++) {
    transposed.push([]);
    for (let j = 0; j < matrix.length; j++) {
      transposed[i].push(matrix[j][i]);
    }
  }
  return transposed;
}
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
  EMBEDDING_MATRIX,
  UNEMBEDDING_MATRIX,
  embedTokenNumber,
  embedInput,
  transposeArray
};
