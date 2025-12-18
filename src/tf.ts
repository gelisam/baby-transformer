/// <reference types="@tensorflow/tfjs" />

const tf = (globalThis as any).tf as typeof import('@tensorflow/tfjs');

type Tensor2D = import('@tensorflow/tfjs').Tensor2D;
type Sequential = import('@tensorflow/tfjs').Sequential;

// Function to set the backend and update UI
async function setBackend(backendSelector: HTMLSelectElement) {
  const requestedBackend = backendSelector.value;

  try {
    await tf.setBackend(requestedBackend);
    console.log(`TensorFlow.js backend set to: ${tf.getBackend()}`);
  } catch (error) {
    console.error(`Failed to set backend to ${requestedBackend}:`, error);
  }
}

export { tf, Tensor2D, Sequential, setBackend };
