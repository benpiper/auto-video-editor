## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2026-07-13 - Wrap radio group in semantic fieldset and link inputs to small text
**Learning:** Screen readers need grouped context for radio inputs (`<fieldset>` and `<legend>`) and explicit association between form inputs and their sub-label helper text (`aria-describedby` linking to `<small>` elements).
**Action:** When adding options or processing steps that form a single logical choice, use `<fieldset>` with `<legend>` instead of generic `<div>` containers. Always link helper `<small>` text with the corresponding `<input>` using `aria-describedby`.
