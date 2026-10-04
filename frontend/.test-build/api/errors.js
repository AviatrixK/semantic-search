export class ApiError extends Error {
    status;
    /** Seconds from the Retry-After header (429 responses), if the server sent one. */
    retryAfter;
    constructor(status, message, retryAfter) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.retryAfter = retryAfter;
    }
}
/** The request never got an answer (backend down, offline, DNS...). */
export class NetworkError extends Error {
    constructor(message = 'Network request failed') {
        super(message);
        this.name = 'NetworkError';
    }
}
/** Retry-After is either a number of seconds or an HTTP date. Returns whole seconds, or undefined if unusable. */
export function parseRetryAfter(value, now = Date.now()) {
    if (value === null)
        return undefined;
    const trimmed = value.trim();
    if (trimmed === '')
        return undefined;
    if (/^\d+(\.\d+)?$/.test(trimmed))
        return Math.ceil(Number(trimmed));
    const date = Date.parse(trimmed);
    if (Number.isNaN(date))
        return undefined;
    return Math.max(0, Math.ceil((date - now) / 1000));
}
/** FastAPI errors: `detail` is a string, or a list of {msg} objects for 422 validation errors. */
export function detailMessage(body) {
    if (typeof body !== 'object' || body === null || !('detail' in body))
        return undefined;
    const detail = body.detail;
    if (typeof detail === 'string')
        return detail;
    if (Array.isArray(detail)) {
        const messages = detail
            .map((d) => (typeof d === 'object' && d !== null && 'msg' in d ? String(d.msg) : ''))
            .map((m) => m.replace(/^Value error,\s*/i, ''))
            .filter(Boolean);
        return messages.length ? messages.join(' ') : undefined;
    }
    return undefined;
}
export function formatWait(seconds) {
    const s = Math.max(1, Math.ceil(seconds));
    if (s < 60)
        return `${s} second${s === 1 ? '' : 's'}`;
    const m = Math.floor(s / 60);
    const rest = s % 60;
    const minutes = `${m} minute${m === 1 ? '' : 's'}`;
    return rest === 0 ? minutes : `${minutes} ${rest} second${rest === 1 ? '' : 's'}`;
}
export function rateLimitMessage(retryAfter) {
    return retryAfter === undefined
        ? 'Too many attempts. Please wait a moment and try again.'
        : `Too many attempts. Please wait ${formatWait(retryAfter)} and try again.`;
}
/** User-facing text for any error thrown by the API client. Never shows stack traces or raw JSON. */
export function describeError(err, context) {
    if (err instanceof NetworkError)
        return "Can't reach the server. Check that the backend is running and try again.";
    if (err instanceof ApiError) {
        if (err.status === 429)
            return rateLimitMessage(err.retryAfter);
        if (context === 'upload') {
            if (err.status === 413) {
                const mb = /max (\d+) MB/i.exec(err.message)?.[1]; // backend: "File too large (max 500 MB)"
                return mb ? `This file is too large. The limit is ${mb} MB.` : 'This file is too large.';
            }
            if (err.status === 415)
                return 'Unsupported file type. Upload an MP4, WebM, MOV or MKV video.';
            if (err.status === 400)
                return err.message === 'Empty file' ? 'That file is empty.' : err.message;
            if (err.status === 403)
                return 'Only administrators can upload videos.';
        }
        if ((context === 'reprocess' || context === 'delete') && err.status === 404)
            return 'That video no longer exists.';
        if (context === 'reprocess' && err.status === 409)
            return 'This video is already being processed.';
        if (context === 'job' && err.status === 404)
            return 'This job no longer exists. The video may have been deleted.';
        if ((context === 'stream' || context === 'transcript') && err.status === 404)
            return 'This video is no longer available.';
        // The backend's 503 for /api/ask carries a short, safe, user-facing reason (not configured, busy, model rejected...).
        if (context === 'ask' && err.status === 503 && err.message)
            return err.message;
        if (err.status === 401) {
            return context === 'login' ? 'Incorrect email or password.' : 'Your session has expired. Please log in again.';
        }
        if (err.status === 409 && context === 'register')
            return 'An account with this email already exists.';
        if (err.status === 422)
            return err.message || 'Please check the details you entered.';
        if (err.status === 403)
            return "You don't have permission to do that.";
        if (err.status >= 500)
            return 'Something went wrong on the server. Please try again.';
        return err.message || `Request failed (${err.status}).`;
    }
    return err instanceof Error && err.message ? err.message : 'Something went wrong. Please try again.';
}
