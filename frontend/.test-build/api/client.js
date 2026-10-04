import { createSseParser } from '../lib/sse.js';
import { ApiError, NetworkError, detailMessage, parseRetryAfter } from './errors.js';
const AUTH_PATHS = ['/auth/login', '/auth/register', '/auth/refresh', '/auth/logout'];
const noLock = (fn) => fn();
/**
 * Cross-tab mutex via the Web Locks API. The refresh token is single-use and presenting an already-used one
 * revokes ALL of the user's sessions. All tabs share one refresh cookie, so two tabs refreshing at the same moment
 * would both send the old token and trip that protection. Under the lock the second tab starts only after the first
 * has stored the rotated cookie, so it sends the new token. Falls back to no lock where the API is unavailable.
 */
export function webLocksRunner(name = 'svs-auth-refresh') {
    return (fn) => {
        const locks = globalThis.navigator?.locks;
        return locks ? locks.request(name, fn) : fn();
    };
}
function abortError() {
    return new DOMException('Aborted', 'AbortError');
}
/** Builds a fetch-style Response from a finished XHR so uploads share the error/parse code with request(). */
function xhrToResponse(xhr) {
    const headers = new Headers();
    for (const line of xhr.getAllResponseHeaders().trim().split(/[\r\n]+/)) {
        const i = line.indexOf(':');
        if (i > 0)
            headers.append(line.slice(0, i).trim(), line.slice(i + 1).trim());
    }
    const noBody = xhr.status === 204 || xhr.status === 205 || xhr.status === 304;
    return new Response(noBody ? null : xhr.responseText, { status: xhr.status, headers });
}
export function createApiClient(options = {}) {
    const baseUrl = options.baseUrl ?? '';
    const doFetch = options.fetch ?? ((input, init) => fetch(input, init));
    const newXhr = options.xhr ?? (() => new XMLHttpRequest());
    const runExclusive = options.runExclusive ?? noLock;
    let token = null;
    let inflight = null;
    const tokenListeners = new Set();
    const expiredListeners = new Set();
    function setToken(next) {
        if (next === token)
            return;
        token = next;
        tokenListeners.forEach((cb) => cb(next));
    }
    function expireSession() {
        setToken(null);
        expiredListeners.forEach((cb) => cb());
    }
    async function send(path, o, accessToken) {
        const headers = { Accept: 'application/json', ...o.headers };
        if (accessToken)
            headers.Authorization = `Bearer ${accessToken}`;
        let body = o.body;
        if (o.json !== undefined) {
            headers['Content-Type'] = 'application/json';
            body = JSON.stringify(o.json);
        }
        try {
            return await doFetch(baseUrl + path, {
                method: o.method ?? (body === undefined ? 'GET' : 'POST'),
                headers,
                body,
                credentials: 'include',
                signal: o.signal,
            });
        }
        catch (e) {
            if (e instanceof DOMException && e.name === 'AbortError')
                throw e;
            throw new NetworkError();
        }
    }
    function sendXhr(path, form, accessToken, o) {
        return new Promise((resolve, reject) => {
            if (o.signal?.aborted)
                return reject(abortError());
            const xhr = newXhr();
            xhr.open('POST', baseUrl + path);
            xhr.withCredentials = true;
            xhr.setRequestHeader('Accept', 'application/json');
            if (accessToken)
                xhr.setRequestHeader('Authorization', `Bearer ${accessToken}`);
            // No Content-Type: the browser must set multipart/form-data with its boundary itself.
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable)
                    o.onProgress?.(e.loaded, e.total);
            };
            xhr.onload = () => resolve(xhrToResponse(xhr));
            xhr.onerror = () => reject(new NetworkError());
            xhr.ontimeout = () => reject(new NetworkError('The upload timed out'));
            xhr.onabort = () => reject(abortError());
            o.signal?.addEventListener('abort', () => xhr.abort(), { once: true });
            xhr.send(form);
        });
    }
    async function toApiError(res) {
        let parsed;
        try {
            parsed = await res.json();
        }
        catch {
            parsed = undefined;
        }
        const message = detailMessage(parsed) ?? (res.statusText || `HTTP ${res.status}`);
        return new ApiError(res.status, message, parseRetryAfter(res.headers.get('Retry-After')));
    }
    async function doRefresh() {
        const res = await send('/auth/refresh', { method: 'POST' }, null);
        if (res.ok) {
            const data = (await res.json());
            setToken(data.access_token);
            return data.access_token;
        }
        if (res.status === 401) {
            setToken(null);
            return null; // no cookie, expired, or revoked: genuinely logged out
        }
        throw await toApiError(res); // 5xx etc.: the session may still be fine, so do not log the user out
    }
    function refresh() {
        // Single flight: concurrent callers (parallel 401s, React StrictMode's double effect) share one request.
        if (!inflight) {
            inflight = runExclusive(doRefresh).finally(() => {
                inflight = null;
            });
        }
        return inflight;
    }
    /** Sends with the current token; on 401 refreshes once and retries once; throws ApiError for any non-2xx. */
    async function withAuthRetry(path, sender) {
        const sentWith = token;
        let res = await sender(sentWith);
        if (res.status === 401 && !AUTH_PATHS.includes(path)) {
            // If another request refreshed while ours was in flight, just retry with the newer token.
            const fresh = token !== null && token !== sentWith ? token : await refresh();
            if (!fresh) {
                expireSession();
                throw new ApiError(401, 'Your session has expired. Please log in again.');
            }
            res = await sender(fresh);
            if (res.status === 401) {
                expireSession();
                throw await toApiError(res);
            }
        }
        if (!res.ok)
            throw await toApiError(res);
        return res;
    }
    async function parse(res) {
        if (res.status === 204)
            return undefined;
        return (res.headers.get('Content-Type') ?? '').includes('json') ? (await res.json()) : undefined;
    }
    return {
        getToken: () => token,
        setToken,
        onTokenChange(cb) {
            tokenListeners.add(cb);
            return () => tokenListeners.delete(cb);
        },
        onSessionExpired(cb) {
            expiredListeners.add(cb);
            return () => expiredListeners.delete(cb);
        },
        refresh,
        async request(path, o = {}) {
            return parse(await withAuthRetry(path, (t) => send(path, o, t)));
        },
        async upload(path, form, o = {}) {
            return parse(await withAuthRetry(path, (t) => sendXhr(path, form, t, o)));
        },
        async stream(path, o) {
            const res = await withAuthRetry(path, (t) => send(path, { headers: { Accept: 'text/event-stream' }, signal: o.signal }, t));
            if (!res.body)
                throw new NetworkError('This browser cannot read streamed responses');
            const reader = res.body.getReader();
            const decoder = new TextDecoder(); // stream: true below, so a character split across chunks is not corrupted
            const parser = createSseParser(o.onEvent);
            try {
                for (;;) {
                    const { done, value } = await reader.read();
                    if (done)
                        break;
                    parser.push(decoder.decode(value, { stream: true }));
                }
                parser.push(decoder.decode());
                parser.end();
            }
            catch (e) {
                if (e instanceof DOMException && e.name === 'AbortError')
                    throw e;
                throw new NetworkError('The connection was interrupted');
            }
        },
    };
}
