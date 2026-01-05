function showError(message: string, toaster: HTMLElement): void {
  toaster.textContent = message;
  toaster.style.display = 'block';

  setTimeout(() => {
    toaster.style.display = 'none';
  }, 3000);
}

export { showError };
