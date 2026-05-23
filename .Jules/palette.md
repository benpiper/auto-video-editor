## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2026-05-23 - [Accessible Form Inputs and Groupings]
**Learning:** Using `aria-describedby` to link inputs to `<small>` helper text, and grouping related radio buttons with semantic `<fieldset>` and `<legend>` tags ensures comprehensive announcements and context by screen readers, which is crucial for complex configuration forms like the video processing parameters.
**Action:** Always link helper text explicitly to form inputs via ID references and use semantic HTML groupings for sets of interactive choices instead of generic `div`s.
