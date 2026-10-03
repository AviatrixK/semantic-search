export function initialQueue() {
    return { items: [] };
}
function update(state, id, allowed, patch) {
    let changed = false;
    const items = state.items.map((item) => {
        if (item.id !== id || !allowed.includes(item.phase))
            return item;
        changed = true;
        return { ...item, ...patch(item) };
    });
    return changed ? { items } : state; // unchanged state keeps React from re-rendering
}
export function queueReducer(state, action) {
    switch (action.type) {
        case 'enqueue':
            return {
                items: [
                    ...state.items,
                    ...action.entries.map((e) => ({
                        id: e.id, title: e.title, fileName: e.file.name, fileSize: e.file.size, file: e.file,
                        phase: 'queued', loaded: 0, total: e.file.size, videoId: null, jobId: null, error: null,
                    })),
                ],
            };
        case 'start': // idempotent: only a queued item can start (StrictMode may fire the effect twice)
            return update(state, action.id, ['queued'], () => ({ phase: 'uploading', loaded: 0 }));
        case 'progress':
            return update(state, action.id, ['uploading'], () => ({ loaded: action.loaded, total: action.total }));
        case 'uploaded':
            return update(state, action.id, ['uploading'], (i) => ({
                phase: 'processing', file: null, loaded: i.total, videoId: action.videoId, jobId: action.jobId,
            }));
        case 'failed':
            return update(state, action.id, ['queued', 'uploading'], () => ({ phase: 'error', file: null, error: action.message }));
        case 'remove':
            return { items: state.items.filter((i) => i.id !== action.id) };
        case 'removeByVideo':
            return { items: state.items.filter((i) => i.videoId !== action.videoId) };
        case 'track':
            return {
                items: [
                    ...state.items,
                    {
                        id: action.id, title: action.title, fileName: action.title, fileSize: 0, file: null, phase: 'processing',
                        loaded: 0, total: 0, videoId: action.videoId, jobId: action.jobId, error: null,
                    },
                ],
            };
    }
}
/** Id of the next file to upload: the first queued one, but only while nothing else is uploading. */
export function nextToStart(state) {
    if (state.items.some((i) => i.phase === 'uploading'))
        return undefined;
    return state.items.find((i) => i.phase === 'queued')?.id;
}
/** True while files are waiting or being sent (closing the tab would lose them). */
export function hasPendingUploads(state) {
    return state.items.some((i) => i.phase === 'queued' || i.phase === 'uploading');
}
