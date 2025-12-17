/// <reference types="@tensorflow/tfjs" />

const tf = (globalThis as any).tf as typeof import('@tensorflow/tfjs');

type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;
type Logs = import('@tensorflow/tfjs').Logs;
type Tensor = import('@tensorflow/tfjs').Tensor;

export { tf, Tensor2D, Sequential, Logs, Tensor };
