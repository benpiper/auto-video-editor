## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2025-10-18 - Form Semantics and Screen Reader Accessibility
**Learning:** Native `aria-describedby` attributes are required to link input fields directly to their respective help text (e.g., `<small>`). Without this, screen readers will not announce the helper text when the input receives focus. Additionally, grouping related radio buttons within a generic `div` instead of a semantic `<fieldset>` breaks screen reader context grouping.
**Action:** When adding helper text to form inputs, always explicitly link them using `aria-describedby` on the input element and assign the matching `id` to the helper text element. Furthermore, when grouping inputs like radio buttons, use `<fieldset>` and `<legend>` tags (styled inline to match visual specs if necessary, e.g., `border: none; padding: 0; margin: 0;`) to preserve semantic structure and proper grouping announcements.
