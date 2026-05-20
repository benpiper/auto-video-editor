## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2026-05-20 - Grouping Related Form Controls
**Learning:** Related sets of form controls, such as radio button groups, can lack context when screen readers announce them individually. Additionally, helper text using `<small>` is often skipped unless explicitly tied to the input it describes.
**Action:** When building forms, wrap related inputs (like radio options for background removal methods) in a semantic `<fieldset>` with a descriptive `<legend>`. Furthermore, explicitly link auxiliary text to inputs using the `aria-describedby` attribute to ensure screen readers announce all contextual information.
