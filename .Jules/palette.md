## 2024-05-18 - Keyboard Accessible File Upload Area
**Learning:** Custom interactive elements (like divs used as upload dropzones) require explicit tabindex, ARIA roles, and keyboard event handlers (Enter/Space) to be fully accessible, unlike native buttons or inputs.
**Action:** Always add `tabindex="0"`, `role="button"`, `:focus-visible` styles, and keydown listeners when converting non-interactive elements into primary UI controls.
