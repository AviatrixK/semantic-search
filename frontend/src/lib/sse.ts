export interface SseEvent {
  /** The `event:` field; "message" when the server sent none. */
  event: string
  /** The `data:` lines joined with "\n". */
  data: string
  id?: string
}

/**
 * Incremental Server-Sent Events parser (the WHATWG rules). Feed it text as it arrives, in chunks cut anywhere: in the middle
 * of a line, between a "\r" and its "\n", between two bytes of one character (decode with TextDecoder `stream: true`). It calls
 * `onEvent` for each complete event. We use fetch + ReadableStream instead of EventSource because EventSource cannot send an
 * Authorization header.
 *
 *   field: value   one field per line ("event", "data", "id"; "retry" is ignored)
 *   : comment      ignored (the server uses these as keep-alives)
 *   blank line     ends the event
 */
export function createSseParser(onEvent: (e: SseEvent) => void): { push(chunk: string): void; end(): void } {
  let buffer = ''
  let event = ''
  let data: string[] = []
  let id: string | undefined

  function line(l: string) {
    if (l === '') {
      if (data.length) onEvent({ event: event || 'message', data: data.join('\n'), ...(id !== undefined ? { id } : {}) })
      event = ''
      data = []
      id = undefined
      return
    }
    if (l.startsWith(':')) return
    const i = l.indexOf(':')
    const field = i < 0 ? l : l.slice(0, i)
    let value = i < 0 ? '' : l.slice(i + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
    else if (field === 'id') id = value
  }

  function drain(final: boolean) {
    for (;;) {
      const m = /\r\n|\n|\r/.exec(buffer)
      if (!m) return
      // A "\r" at the very end may be the first half of "\r\n": wait for the next chunk to find out.
      if (m[0] === '\r' && m.index + 1 === buffer.length && !final) return
      line(buffer.slice(0, m.index))
      buffer = buffer.slice(m.index + m[0].length)
    }
  }

  return {
    push(chunk) {
      buffer += chunk
      drain(false)
    },
    end() {
      drain(true) // an event that never got its blank line is dropped, as the spec says
      buffer = ''
      event = ''
      data = []
      id = undefined
    },
  }
}
