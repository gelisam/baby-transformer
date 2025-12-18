import { Tensor2D } from "./tf.js";
import { TrainingData } from "./types.js";

/**
 * Manages the lifecycle of TensorFlow.js tensors to prevent memory leaks.
 * 
 * This class provides methods to track, dispose, and manage tensor resources
 * throughout the application lifecycle.
 */
class ResourceManager {
  private resources: Set<Tensor2D> = new Set();

  /**
   * Register a tensor for tracking.
   * @param tensor - The tensor to track
   * @returns The same tensor for chaining
   */
  track(tensor: Tensor2D): Tensor2D {
    this.resources.add(tensor);
    return tensor;
  }

  /**
   * Safely dispose a single tensor.
   * Handles cases where the tensor is undefined or already disposed.
   * @param tensor - The tensor to dispose
   */
  dispose(tensor: Tensor2D | undefined): void {
    if (tensor && !tensor.isDisposed) {
      try {
        tensor.dispose();
        this.resources.delete(tensor);
      } catch (e) {
        // Tensor may already be disposed
        console.warn('Failed to dispose tensor:', e);
      }
    }
  }

  /**
   * Safely dispose training data tensors.
   * @param data - The training data containing tensors to dispose
   */
  disposeTrainingData(data: TrainingData | undefined): void {
    if (data) {
      this.dispose(data.inputTensor);
      this.dispose(data.outputTensor);
    }
  }

  /**
   * Dispose all tracked resources.
   * Useful for cleanup when the application is shutting down.
   */
  disposeAll(): void {
    for (const tensor of this.resources) {
      if (!tensor.isDisposed) {
        try {
          tensor.dispose();
        } catch (e) {
          console.warn('Failed to dispose tracked tensor:', e);
        }
      }
    }
    this.resources.clear();
  }
}

export { ResourceManager };
