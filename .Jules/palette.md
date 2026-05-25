## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2025-05-18 - Form Semantics and Screen Readers
**Learning:** Generic `div`s and `label`s to wrap radio button groups or using adjacent `<small>` tags for helper text do not provide proper context for screen readers. They will often miss the helper text and group context entirely.
**Action:** Always link helper text explicitly with `aria-describedby` pointing to the helper text's ID. When grouping radio buttons, strictly wrap them in a `<fieldset>` with a semantic `<legend>`, and use CSS (`border: none; padding: 0;`) to maintain visual design while improving structural accessibility.
