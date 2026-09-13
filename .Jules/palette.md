## 2025-02-12 - Custom Interactive Elements Accessibility
**Learning:** Custom interactive elements (like `div`s acting as buttons or dropzones) do not receive keyboard focus or semantics by default. This makes them inaccessible to keyboard and screen reader users.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, an appropriate `role` (e.g., `role="button"`), an `aria-label`, bind `keydown` event listeners for 'Enter' and 'Space', and provide `:focus-visible` styling matching the design system.

## 2025-05-18 - Dynamic UI State Context Loss
**Learning:** Dynamic DOM section swaps (like hiding a progress bar and showing a completion section) cause loss of context and leave keyboard/screen reader users stranded without an anchor point.
**Action:** When dynamically swapping major UI sections, explicitly move focus to a newly revealed heading by adding `tabindex="-1"` and calling `.focus()`. Furthermore, use native ARIA roles (e.g. `role="progressbar"`) and `aria-live="polite"` to ensure progress transitions are accessible.
## 2026-09-13 - Added visual disabled states to buttons
**Learning:** Found that buttons had `disabled=true` set via JS during async operations, but lacked CSS to visually communicate this state to users. Without a `:disabled` style, buttons still triggered hover animations (like transform and box-shadow) while inactive, which is confusing and poor UX.
**Action:** Always pair JS `disabled` states with CSS `:disabled` visual styles (e.g., opacity, `cursor: not-allowed`). Restrict hover animations using `:not(:disabled):hover` instead of `pointer-events: none` to preserve the `not-allowed` cursor.
