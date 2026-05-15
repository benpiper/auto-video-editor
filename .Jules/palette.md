## 2024-05-15 - Make custom div buttons accessible
**Learning:** The custom `div` used for file uploading lacks standard button accessibility. Users relying on keyboard navigation cannot focus on or interact with it to trigger the file input dialog.
**Action:** Always ensure custom interactive elements like `div`s functioning as buttons have `tabindex="0"`, `role="button"`, appropriate `aria-label`s, `:focus-visible` states, and handle `keydown` events for "Enter" and "Space" explicitly to maintain keyboard accessibility.
