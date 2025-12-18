import { Tensor2D } from "./tf.js";
import { TrainingData } from "./types.js";

/**
 * Manages the lifecycle of TensorFlow.js tensors to prevent memory leaks
 */
class ResourceManager {
  private resources: Set<Tensor2D> = new Set();

  /**
   * Register a tensor for tracking
   */
  track(tensor: Tensor2D): Tensor2D {
    this.resources.add(tensor);
    return tensor;
  }

  /**
   * Safely dispose a single tensor
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
   * Safely dispose training data tensors
   */
  disposeTrainingData(data: TrainingData | undefined): void {
    if (data) {
      this.dispose(data.inputTensor);
      this.dispose(data.outputTensor);
    }
  }

  /**
   * Dispose all tracked resources
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
