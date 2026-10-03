/** Presigned video URLs expire (the backend signs them for 1 hour); reuse one for 50 minutes, never a stale one. */
export const STREAM_URL_TTL_MS = 50 * 60 * 1000;
export function createStreamUrlCache(fetchUrl, options = {}) {
    const ttl = options.ttlMs ?? STREAM_URL_TTL_MS;
    const now = options.now ?? Date.now;
    const done = new Map();
    const pending = new Map();
    return {
        get(videoId) {
            const hit = done.get(videoId);
            if (hit && now() - hit.at < ttl)
                return Promise.resolve(hit.url);
            const inflight = pending.get(videoId);
            if (inflight)
                return inflight;
            const request = fetchUrl(videoId).then((url) => {
                pending.delete(videoId);
                done.set(videoId, { url, at: now() });
                return url;
            }, (error) => {
                pending.delete(videoId); // failures are never cached
                throw error;
            });
            pending.set(videoId, request);
            return request;
        },
        invalidate(videoId) {
            done.delete(videoId);
            pending.delete(videoId);
        },
        clear() {
            done.clear();
            pending.clear();
        },
    };
}
