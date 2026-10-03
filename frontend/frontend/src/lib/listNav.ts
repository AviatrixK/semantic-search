/** Keyboard navigation through a results list while focus stays in the search box (combobox pattern). */
export function moveActive(current: number, key: 'ArrowDown' | 'ArrowUp', count: number): number {
  if (count <= 0) return -1
  if (key === 'ArrowDown') return current < 0 ? 0 : Math.min(current + 1, count - 1)
  return current < 0 ? count - 1 : Math.max(current - 1, 0) // from "nothing selected", Up goes to the last result
}

/**
 * Scroll position that brings an element fully into view inside a scrollable box, or null if it already is.
 * (Scrolling the box directly, instead of scrollIntoView, never drags the whole page along.)
 */
export function scrollTargetFor(
  boxScrollTop: number, boxHeight: number, elTop: number, elHeight: number, pad = 8,
): number | null {
  if (elTop < boxScrollTop + pad) return Math.max(0, elTop - pad)
  if (elTop + elHeight > boxScrollTop + boxHeight - pad) return elTop + elHeight - boxHeight + pad
  return null
}
