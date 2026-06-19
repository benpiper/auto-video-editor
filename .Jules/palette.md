## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2025-10-24 - Form Semantics and Helper Text Associations
**Learning:** Screen readers do not automatically announce helper text (like `<small>` elements) adjacent to inputs, and generic `div` wrappers around radio button groups fail to convey structural grouping, leaving users without context.
**Action:** When building HTML forms, always link helper text to their respective inputs using the `aria-describedby` attribute, and wrap related inputs (like radio groups) in a `<fieldset>` with a semantic `<legend>` to ensure proper grouping and comprehensive announcements by screen readers.
