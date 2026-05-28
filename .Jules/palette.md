## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2025-05-19 - Form Semantics for Complex Inputs and Radio Groups
**Learning:** Generic `<div>` elements and simple `<label>` tags provide poor semantics for related form elements (like radio groups) or inputs with complex helper text, leading to confusing announcements by screen readers.
**Action:** When building HTML forms, always link helper text (like `<small>` elements) to their respective inputs using the `aria-describedby` attribute. For related inputs like radio groups, wrap them in a `<fieldset>` with a semantic `<legend>`. Apply inline CSS (e.g., `border: none; padding: 0;`) to these native elements to maintain the original visual design system while dramatically improving structural context for accessibility.
