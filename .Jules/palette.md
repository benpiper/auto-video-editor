## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2025-05-19 - Semantic Form Grouping
**Learning:** Using generic `div` and `label` tags for grouping radio buttons creates a disconnected experience for screen reader users, who won't hear the group context when tabbing between options.
**Action:** Always wrap related radio button groups in a `<fieldset>` and provide a semantic `<legend>`. To maintain existing visual design patterns, apply inline CSS (`border: none; padding: 0; margin: 0;`) to the `<fieldset>`.

## 2025-05-19 - Associating Helper Text
**Learning:** Form helper text (often found in `<small>` tags below inputs) is ignored by screen readers when focusing on the input field if it is not explicitly linked.
**Action:** Link all supplementary description or helper text to their corresponding form inputs using the `aria-describedby` attribute pointing to the ID of the helper text element.
