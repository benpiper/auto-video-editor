
## 2026-05-12 - Making Custom Upload Dropzones Accessible
**Learning:** In custom interactive elements like a `div`-based drag-and-drop zone (`.upload-area`), visual styling (borders, hover states) is not enough. Screen readers skip them, and keyboard users cannot reach them without explicit semantics (`role="button"`), tab ordering (`tabindex="0"`), descriptive context (`aria-label`), explicit focus rings (`:focus-visible`), and JS keydown bindings for 'Enter'/'Space'.
**Action:** Always verify keyboard accessibility manually or programmatically. When using non-semantic HTML elements for interaction, always apply the full suite of ARIA properties and keyboard event listeners to ensure parity with native HTML elements like `<button>` or `<input>`.
