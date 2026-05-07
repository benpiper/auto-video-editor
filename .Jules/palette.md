
## 2024-05-20 - Custom Dropzone Keyboard Accessibility
**Learning:** Custom 'div'-based file dropzones often miss native keyboard interactions because they lack semantic meaning and focusability. Simply making them clickable and drag-and-droppable leaves keyboard-only and screen-reader users unable to activate the file picker dialog.
**Action:** When creating or modifying custom interactive elements (like `div`s acting as buttons), explicitly add `:focus-visible` styles matching the primary theme, ensure `tabindex="0"` and `role="button"` are set, and bind `keydown` listeners for 'Enter' and 'Space' alongside standard 'click' events to ensure full keyboard accessibility.
