## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2025-07-04 - Form Grouping Semantics
**Learning:** Radio button groups wrapped in generic `div` tags lose context for screen reader users when they tab through the options, as the visual group label is not semantically linked.
**Action:** Always wrap related form inputs, especially radio button groups, in a semantic `<fieldset>` with a descriptive `<legend>`. Ensure any inline styles (like overriding margins) are ordered correctly (e.g., `margin: 0; margin-top: 1rem;`) so specific styles take precedence over shorthand properties.
