/// <reference types="@tensorflow/tfjs" />

const tf = (globalThis as any).tf as typeof import('@tensorflow/tfjs');

type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;

export { tf, Tensor2D, Sequential };
