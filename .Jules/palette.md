## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2024-05-24 - Semantic Form Grouping and Context
**Learning:** Using generic `<div>` elements and `<label>` tags to group radio buttons prevents screen readers from announcing the group context when users navigate between options. Additionally, standalone `<small>` helper text is not read by screen readers when focusing on the related input.
**Action:** Always wrap radio button groups in a `<fieldset>` with a `<legend>` for proper grouping context, using inline styles like `border: none; padding: 0; margin: 0;` to maintain visual design. Ensure all helper text is explicitly linked to inputs using `aria-describedby` matching the helper text's `id`.
