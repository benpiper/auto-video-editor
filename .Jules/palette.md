## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2025-05-20 - Form Helper Text & Group Accessibility
**Learning:** Screen readers do not automatically associate adjacent helper text (e.g., `<small>`) with form inputs, and disjointed radio buttons lack context without a structural group, leading to confusing auditory navigation.
**Action:** Always link helper text to inputs using the `aria-describedby` attribute, and wrap related radio buttons in a semantic `<fieldset>` with a `<legend>` to ensure proper grouping, using inline CSS like `border: none; padding: 0; margin: 0;` to preserve visual design.
