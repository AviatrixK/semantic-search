import { DEFAULT_FILTERS, filtersKey } from './searchParams.js';
const defaultSchedule = (fn, ms) => {
    const id = setTimeout(fn, ms);
    return () => clearTimeout(id);
};
export const IDLE = { status: 'idle', query: '', filtersKey: '', hits: [], error: null, retryAfter: null };
/**
 * Debounced search with cancellation: typing never queues requests (a new one aborts the previous), an unchanged query
 * is not searched twice, and only the newest response may reach the screen. Previous results stay visible while the
 * next ones load.
 */
export function createSearcher(o) {
    const debounceMs = o.debounceMs ?? 400;
    const minLength = o.minLength ?? 2;
    const schedule = o.schedule ?? defaultSchedule;
    let query = '';
    let filters = o.filters ?? DEFAULT_FILTERS;
    let fkey = filtersKey(filters);
    let state = IDLE;
    let cancelTimer = null;
    let controller = null;
    let runId = 0;
    let disposed = false;
    const emit = (next) => {
        state = next;
        if (!disposed)
            o.onState(next);
    };
    const stopPending = () => {
        cancelTimer?.();
        cancelTimer = null;
        controller?.abort();
        controller = null;
        runId++; // invalidates a response that is still on its way
    };
    async function execute() {
        cancelTimer = null;
        const q = query;
        if (q.length < minLength)
            return;
        if (state.query === q && state.filtersKey === fkey && (state.status === 'ready' || state.status === 'loading'))
            return; // already showing / fetching
        controller?.abort();
        const mine = ++runId;
        controller = new AbortController();
        emit({ ...state, status: 'loading', query: q, filtersKey: fkey, error: null, retryAfter: null });
        const used = filters;
        const usedKey = fkey;
        try {
            const hits = await o.run(q, controller.signal, used);
            if (mine !== runId)
                return;
            emit({ status: 'ready', query: q, filtersKey: usedKey, hits, error: null, retryAfter: null });
        }
        catch (error) {
            if (mine !== runId)
                return; // aborted or superseded
            const d = o.describe(error);
            emit({ status: 'error', query: q, filtersKey: usedKey, hits: state.hits, error: d.message, retryAfter: d.retryAfter ?? null });
        }
    }
    return {
        setQuery(raw) {
            const q = raw.trim().replace(/\s+/g, ' ');
            if (q === query)
                return;
            query = q;
            cancelTimer?.();
            cancelTimer = null;
            if (q.length < minLength) {
                stopPending();
                emit(IDLE);
                return;
            }
            if (state.query === q && state.filtersKey === fkey && state.status === 'ready') {
                stopPending();
                return;
            }
            cancelTimer = schedule(() => void execute(), debounceMs);
        },
        submit() {
            cancelTimer?.();
            cancelTimer = null;
            if (state.query === query && state.status === 'error')
                state = { ...state, status: 'idle', query: '' }; // retry after an error
            void execute();
        },
        setFilters(next) {
            const key = filtersKey(next);
            if (key === fkey)
                return;
            filters = next;
            fkey = key;
            if (query.length < minLength)
                return; // nothing to search yet: the filters apply to the next query
            cancelTimer?.();
            cancelTimer = null;
            void execute();
        },
        dispose() {
            disposed = true;
            stopPending();
        },
    };
}
