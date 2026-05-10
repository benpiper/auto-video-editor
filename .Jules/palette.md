## 2026-05-10 - Custom Interactive Elements Missing Semantic HTML
**Learning:** The custom `div` element used as an upload dropzone lacked native keyboard interactivity and screen reader semantics, which is a common accessibility trap when replacing standard `<input>` or `<button>` tags with styled `div`s.
**Action:** When implementing custom interactive `div`s, always explicitly map them to standard interactive element behaviors by adding `tabindex="0"`, `role="button"`, `:focus-visible` styles, and binding `keydown` listeners for 'Enter' and 'Space' keys.
