/** mm:ss. Minutes are not wrapped into hours (a 65 minute video is "65:12"). "—" when unknown. */
export function formatDuration(totalSeconds) {
    if (totalSeconds === null || totalSeconds === undefined || !Number.isFinite(totalSeconds) || totalSeconds < 0)
        return '—';
    const s = Math.floor(totalSeconds);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
export function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0)
        return '—';
    if (bytes < 1024)
        return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let value = bytes / 1024;
    let i = 0;
    while (value >= 1024 && i < units.length - 1) {
        value /= 1024;
        i++;
    }
    return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`;
}
/** Local date and time, e.g. "Mar 4, 2026, 05:06 AM". `locale` / `timeZone` exist so tests are deterministic. */
export function formatDate(iso, locale, timeZone) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime()))
        return '—';
    return d.toLocaleString(locale, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone });
}
const STAGE_LABELS = {
    queued: 'Waiting in queue',
    extracting: 'Extracting audio',
    transcribing: 'Transcribing speech',
    embedding: 'Indexing for search',
    done: 'Ready',
    failed: 'Failed',
};
export function stageLabel(stage) {
    return STAGE_LABELS[stage] ?? stage;
}
export function isTerminalStage(stage) {
    return stage === 'done' || stage === 'failed';
}
/** Video.status: "uploaded" really means "stored, waiting for or in processing". */
export function statusBadge(status) {
    switch (status) {
        case 'ready':
            return { label: 'Ready', tone: 'ok' };
        case 'uploaded':
            return { label: 'Processing', tone: 'busy' };
        case 'failed':
            return { label: 'Failed', tone: 'bad' };
        default:
            return { label: status, tone: 'neutral' };
    }
}
