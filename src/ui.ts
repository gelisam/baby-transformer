// Function to display toaster-style error messages
function showError(message: string): void {
  const toaster = document.getElementById('toaster')!;
  toaster.textContent = message;
  toaster.style.display = 'block';

  // Hide the toaster after 3 seconds
  setTimeout(() => {
    toaster.style.display = 'none';
  }, 3000);
}

export { showError };
