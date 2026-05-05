## 2024-05-05 - Divs as Buttons Require Full Polyfill
**Learning:** When using a `div` element as an interactive button (e.g., a file upload dropzone), simply adding `cursor: pointer` and a `click` listener is insufficient for accessibility. It completely excludes keyboard and screen reader users.
**Action:** Always provide a complete "button polyfill" for non-interactive elements: add `role="button"`, `tabindex="0"`, an appropriate `aria-label`, bind `keydown` for Enter/Space keys, and include `:focus-visible` styling.
