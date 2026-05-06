## 2024-05-18 - Keyboard bindings for custom elements
**Learning:** Custom interactive elements (like `div` elements used as buttons or drop zones) do not natively receive keyboard focus or events. It's insufficient to just add a `click` event listener.
**Action:** When creating custom interactive elements, always explicitly add `tabindex="0"`, `role="button"`, proper ARIA attributes, a `keydown` listener handling 'Enter' and 'Space', and a visible `:focus-visible` state matching the theme.
