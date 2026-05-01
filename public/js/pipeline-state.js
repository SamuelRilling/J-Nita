let activeStage = 'input';
const completedStages = new Set();

export function getActiveStage() { return activeStage; }

export function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

export function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function setActiveStage(stage) {
    activeStage = stage;
    document.getElementById('pipeline').dataset.active = stage;

    ['input', 'process', 'result'].forEach(s => {
        const col = document.getElementById(`col-${s}`);
        const tab = document.getElementById(`tab-${s}`);
        const seg = document.querySelector(`.stripe-seg[data-seg="${s}"]`);
        col.classList.toggle('is-active', s === stage);
        col.classList.toggle('is-closed', s !== stage);
        tab.tabIndex = s !== stage ? 0 : -1;
        if (seg) {
            seg.classList.toggle('is-active', s === stage);
            seg.classList.toggle('was-active', completedStages.has(s) && s !== stage);
        }
    });
}

export function triggerStageComplete(stage) {
    return new Promise(async resolve => {
        const col = document.getElementById(`col-${stage}`);
        if (col.classList.contains('was-completed')) { resolve(); return; }

        if (!prefersReducedMotion()) {
            col.classList.add('showing-complete-banner');
            await wait(1800);
            col.classList.remove('showing-complete-banner');
            col.classList.add('showing-complete-check');
            await wait(800);
            col.classList.remove('showing-complete-check');
        }

        col.classList.add('was-completed');
        completedStages.add(stage);
        resolve();
    });
}

export function resetPipelineState() {
    completedStages.clear();
    activeStage = 'input';
    ['input', 'process', 'result'].forEach(s => {
        document.getElementById(`col-${s}`)
            .classList.remove('was-completed', 'showing-complete-banner', 'showing-complete-check');
    });
}
