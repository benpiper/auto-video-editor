## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2026-06-30 - Added accessible `aria-describedby` and `<fieldset>` for forms
**Learning:** Screen readers often miss `<small>` helper text or fail to announce radio group relationships if they are simply wrapped in `div`s. We have to specifically link helper text using `aria-describedby` and group related inputs with `<fieldset>` and `<legend>`.
**Action:** Next time forms have instructional `<small>` tags, always link them using `aria-describedby`. Next time there is a radio group, always use `<fieldset>` and `<legend>` for proper semantics.
