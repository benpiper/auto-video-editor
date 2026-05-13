## 2026-05-13 - Keyboard Accessible Custom Dropzones
**Learning:** Custom 'div'-based file dropzones are often inaccessible to keyboard users because they lack inherent focusability and click semantics.
**Action:** When creating custom interactive elements like dropzones, always add `tabindex="0"`, an appropriate `role` (like "button"), an `aria-label`, bind `keydown` listeners for 'Enter' and 'Space', and include `:focus-visible` styles.
