## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2025-06-23 - Form Control Grouping & Helper Text
**Learning:** Grouping related form controls (like radio buttons) inside generic `div`s with pseudo-labels, or placing helper text near inputs without programmatic association, causes screen readers to misinterpret form structure and miss critical context.
**Action:** Always wrap related input groups (e.g., radio buttons) in a `<fieldset>` with a semantic `<legend>`, applying `border: none; padding: 0; margin: 0;` to maintain visual design. Link helper text (like `<small>`) to inputs using `aria-describedby`.
