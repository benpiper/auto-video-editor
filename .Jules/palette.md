## 2024-04-30 - Custom Upload Area Keyboard Accessibility
**Learning:** Custom div-based upload areas (`.upload-area`) often lack native keyboard focus and interaction support. While click/drag-and-drop are handled, keyboard users cannot trigger the hidden file input.
**Action:** When using a non-interactive element as an interactive dropzone, always add `tabindex="0"`, `role="button"`, an `aria-label`, a `keydown` listener for Enter/Space, and `:focus-visible` styling to ensure parity with mouse interactions.
