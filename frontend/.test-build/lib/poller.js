import { ApiError } from '../api/errors.js';
export const defaultSleep = (ms, signal) => new Promise((resolve) => {
    if (signal.aborted)
        return resolve();
    const id = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(id); resolve(); }, { once: true });
});
/** Delay before the next attempt: the normal interval, a longer wait after errors, or the server's Retry-After on 429. */
export function nextDelay(interval, maxBackoff, consecutiveErrors, error) {
    if (consecutiveErrors === 0)
        return interval;
    if (error instanceof ApiError && error.status === 429 && error.retryAfter)
        return Math.max(interval, error.retryAfter * 1000);
    return Math.min(maxBackoff, interval * 2 ** consecutiveErrors);
}
/**
 * Polls one request at a time (the next poll is scheduled only after the previous one finished, so requests never
 * pile up on a slow server). Returns a function that stops it; also stops after isDone or a fatal error.
 */
export function startPolling(o) {
    const controller = new AbortController();
    const { signal } = controller;
    const interval = o.intervalMs ?? 2000;
    const maxBackoff = o.maxBackoffMs ?? 10000;
    const sleep = o.sleep ?? defaultSleep;
    void (async () => {
        let consecutive = 0;
        while (!signal.aborted) {
            let delay = interval;
            try {
                const value = await o.fetch(signal);
                if (signal.aborted)
                    return;
                consecutive = 0;
                o.onUpdate(value);
                if (o.isDone(value))
                    return;
            }
            catch (error) {
                if (signal.aborted)
                    return;
                consecutive++;
                const fatal = o.isFatal?.(error) ?? false;
                o.onError(error, { consecutive, fatal });
                if (fatal)
                    return;
                delay = nextDelay(interval, maxBackoff, consecutive, error);
            }
            await sleep(delay, signal);
        }
    })();
    return () => controller.abort();
}
