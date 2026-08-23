## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.

## 2026-08-23 - Form Semantics for Radio Button Groups
**Learning:** Using a generic `<div>` and `<label>` to group a set of radio buttons fails to provide structural context to screen readers, meaning users only hear individual radio button options without knowing the overarching group name.
**Action:** When grouping radio buttons, always use a `<fieldset>` with a `<legend>` to properly associate the group label with its options. Apply inline CSS like `border: none; padding: 0; margin: 0;` to maintain the existing visual design system while dramatically improving accessibility.
