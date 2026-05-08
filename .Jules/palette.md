## 2026-05-08 - Keyboard Accessibility for Custom Interactive Divs
**Learning:** Custom interactive elements (like `div.upload-area` used as buttons) must explicitly have `tabindex="0"`, `role="button"`, and listen for Enter/Space `keydown` events. Without these, they are invisible to screen readers and keyboard users.
**Action:** Always ensure full keyboard equivalence and semantic role mappings when transforming non-interactive HTML elements into interactive controls.
