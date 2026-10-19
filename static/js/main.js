document.querySelectorAll("[data-file-label]").forEach(input => {
  input.addEventListener("change", () => {
    const label = input.closest(".upload-zone").querySelector(".file-label");
    label.textContent = input.files && input.files.length ? input.files[0].name : "No file selected";
  });
});
document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => {
    const textarea = document.querySelector(".chat-input textarea");
    if (textarea) {
      textarea.value = button.dataset.question;
      textarea.focus();
    }
  });
});
document.querySelectorAll('input[type="date"]').forEach(input => {
  if (!input.value) input.min = new Date().toISOString().slice(0,10);
});
