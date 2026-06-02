const page = document.body.dataset.page;

if (page === "index") {
    setupPipelinePage();
}

if (page === "results") {
    loadSessionsTable();
}

function setupPipelinePage() {
    const socket = io();
    const modeSelector = document.getElementById("mode-selector");
    const startButton = document.getElementById("start-button");
    const state = {
        currentMode: null,
        isRunning: false,
        counters: {
            identified: 0,
            lost: 0,
            qr_direct: 0,
            qr_enhanced: 0,
            crnn: 0,
        },
    };

    fetch("/api/modes")
        .then((response) => response.json())
        .then((data) => {
            renderModeButtons(data.modes, data.current_mode, data.mode_details, modeSelector, state);
            applyModeAvailability(data.mode_details[data.current_mode]);
        })
        .catch(() => {
            modeSelector.innerHTML = "<p>Failed to load modes.</p>";
        });

    startButton.addEventListener("click", async () => {
        startButton.disabled = true;
        const response = await fetch("/api/start", { method: "POST" });
        if (!response.ok) {
            startButton.disabled = false;
        }
    });

    socket.on("session_started", (payload) => {
        state.isRunning = true;
        resetCounters(state);
        resetBlockStates();
        setRunningControls(true);
        state.currentMode = payload.mode;
        setBlockState("sorting", "running");
    });

    socket.on("session_finished", () => {
        state.isRunning = false;
        setRunningControls(false);
    });

    socket.on("block_update", (payload) => {
        applyBlockUpdate(payload);
    });

    socket.on("counter_update", (payload) => {
        applyCounterUpdate(payload, state);
    });

    function setRunningControls(isRunning) {
        startButton.disabled = isRunning;
        document.querySelectorAll(".mode-button").forEach((button) => {
            button.disabled = isRunning;
        });
    }
}

function renderModeButtons(modes, currentMode, modeDetails, container, state) {
    state.currentMode = currentMode;
    container.innerHTML = "";

    modes.forEach((mode) => {
        const button = document.createElement("button");
        button.className = "mode-button";
        button.type = "button";
        button.textContent = mode;
        if (mode === currentMode) {
            button.classList.add("is-active");
        }

        button.addEventListener("click", async () => {
            const response = await fetch("/api/set_mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode }),
            });

            if (!response.ok) {
                return;
            }

            state.currentMode = mode;
            document.querySelectorAll(".mode-button").forEach((item) => {
                item.classList.toggle("is-active", item.textContent === mode);
            });
            applyModeAvailability(modeDetails[mode]);
        });

        container.appendChild(button);
    });
}

function applyModeAvailability(modeConfig) {
    const mapping = {
        detection: modeConfig.yolo,
        enhanced_qr: modeConfig.enhanced_qr,
        crnn: modeConfig.crnn,
    };

    Object.entries(mapping).forEach(([block, isEnabled]) => {
        const card = document.querySelector(`[data-block="${block}"]`);
        if (!card) {
            return;
        }
        card.classList.toggle("is-disabled", !isEnabled);
    });
}

function resetBlockStates() {
    document.querySelectorAll(".block-card").forEach((card) => {
        card.classList.remove("is-running", "is-done", "is-failed");
    });
    setBlockState("registration", "pending");
    setBlockState("sorting", "pending");
    setBlockState("detection", "pending");
    setBlockState("enhanced_qr", "pending");
    setBlockState("crnn", "pending");
    setBlockState("identified", "pending");
    setBlockState("lost", "pending");
    updateBlockSummary("registration", "Registered: 0 baggage items");
    updateBlockSummary("sorting", "Processing: 0 / 0");
}

function applyBlockUpdate(payload) {
    const { block, status, count, current, total } = payload;
    setBlockState(block, status);

    if (block === "registration" && typeof count === "number") {
        updateBlockSummary(block, `Registered: ${count} baggage items`);
    }

    if (block === "sorting") {
        if (typeof current === "number" && typeof total === "number") {
            updateBlockSummary(block, `Processing: ${current} / ${total}`);
        } else if (typeof count === "number") {
            updateBlockSummary(block, `Processing: ${count} / ${count}`);
        }
    }
}

function setBlockState(block, status) {
    const card = document.querySelector(`[data-block="${block}"]`);
    if (!card) {
        return;
    }

    card.classList.remove("is-running", "is-done", "is-failed");
    if (status === "running") {
        card.classList.add("is-running");
    } else if (status === "done") {
        card.classList.add("is-done");
    } else if (status === "failed") {
        card.classList.add("is-failed");
    }
}

function updateBlockSummary(block, text) {
    const field = document.querySelector(`[data-block="${block}"] [data-field="summary"]`);
    if (field) {
        field.textContent = text;
    }
}

function applyCounterUpdate(payload, state) {
    const { block, delta, method } = payload;
    state.counters[block] += delta;

    if (block === "identified") {
        setBlockState("identified", "done");
        updateBlockSummary("identified", `Total: ${state.counters.identified}`);
        if (method && Object.hasOwn(state.counters, method)) {
            state.counters[method] += delta;
            const methodField = document.querySelector(`[data-method="${method}"]`);
            if (methodField) {
                methodField.textContent = String(state.counters[method]);
            }
        }
    }

    if (block === "lost") {
        setBlockState("lost", "failed");
        updateBlockSummary("lost", `Total: ${state.counters.lost}`);
    }
}

function resetCounters(state) {
    state.counters = {
        identified: 0,
        lost: 0,
        qr_direct: 0,
        qr_enhanced: 0,
        crnn: 0,
    };
    updateBlockSummary("identified", "Total: 0");
    updateBlockSummary("lost", "Total: 0");
    document.querySelectorAll("[data-method]").forEach((field) => {
        field.textContent = "0";
    });
}

async function loadSessionsTable() {
    const body = document.getElementById("sessions-body");
    const expandedSessions = new Set();
    initializeImageModal();

    try {
        const response = await fetch("/api/results/sessions");
        const data = await response.json();

        if (!data.sessions.length) {
            body.innerHTML = '<tr><td colspan="9" class="empty-cell">No sessions recorded yet.</td></tr>';
            return;
        }

        renderSessionRows(data.sessions, body, expandedSessions);
        body.addEventListener("click", async (event) => {
            const toggleButton = event.target.closest("[data-toggle-session]");
            const imageTrigger = event.target.closest("[data-image-url]");

            if (imageTrigger) {
                openImageModal(imageTrigger.dataset.imageUrl);
                return;
            }

            if (!toggleButton) {
                return;
            }

            const sessionId = toggleButton.dataset.toggleSession;
            if (expandedSessions.has(sessionId)) {
                expandedSessions.delete(sessionId);
                renderSessionRows(data.sessions, body, expandedSessions);
                return;
            }

            expandedSessions.add(sessionId);
            renderSessionRows(data.sessions, body, expandedSessions);
            await populateSessionDetails(sessionId, body);
        });
    } catch {
        body.innerHTML = '<tr><td colspan="9" class="empty-cell">Failed to load sessions.</td></tr>';
    }
}

function renderSessionRows(sessions, body, expandedSessions) {
    body.innerHTML = sessions
        .map((session) => {
            const isExpanded = expandedSessions.has(session.session_id);
            return `
                <tr class="session-row">
                    <td>
                        <button
                            class="expand-button"
                            type="button"
                            data-toggle-session="${session.session_id}"
                            aria-expanded="${isExpanded}"
                            aria-label="${isExpanded ? "Collapse" : "Expand"} session ${session.session_id_short}"
                        >
                            ${isExpanded ? "-" : "+"}
                        </button>
                    </td>
                    <td>${session.session_id_short}</td>
                    <td>${session.mode}</td>
                    <td>${session.total}</td>
                    <td>${session.identified}</td>
                    <td>${session.lost}</td>
                    <td>${session.qr_direct}</td>
                    <td>${session.qr_enhanced}</td>
                    <td>${session.crnn}</td>
                </tr>
                ${
                    isExpanded
                        ? `<tr class="session-detail-shell" data-session-detail="${session.session_id}">
                            <td colspan="9" class="session-detail-cell">
                                <div class="session-detail-loading">Loading session details...</div>
                            </td>
                        </tr>`
                        : ""
                }
            `;
        })
        .join("");
}

async function populateSessionDetails(sessionId, body) {
    const row = body.querySelector(`[data-session-detail="${sessionId}"] .session-detail-cell`);
    if (!row) {
        return;
    }

    try {
        const response = await fetch(`/api/results/session/${sessionId}`);
        const data = await response.json();
        row.innerHTML = renderSessionDetailTables(
            data.registration_results || [],
            data.unidentified_images || [],
        );
    } catch {
        row.innerHTML = '<div class="session-detail-loading">Failed to load session details.</div>';
    }
}

function renderSessionDetailTables(registrationResults, unidentifiedImages) {
    return `
        <div class="detail-section">
            <h3 class="detail-section-title">Registration Results</h3>
            ${renderRegistrationResultsTable(registrationResults)}
        </div>
        <div class="detail-section">
            <h3 class="detail-section-title">Unidentified Images</h3>
            ${renderUnidentifiedImagesTable(unidentifiedImages)}
        </div>
    `;
}

function renderRegistrationResultsTable(registrationResults) {
    if (!registrationResults.length) {
        return '<div class="session-detail-loading">No registration results found for this session.</div>';
    }

    return `
        <div class="session-detail-table-wrap">
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>LPN</th>
                        <th>From</th>
                        <th>To</th>
                        <th>Flight</th>
                        <th>Date</th>
                        <th>Passenger</th>
                        <th>Class</th>
                        <th>Pieces</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Method</th>
                        <th>Matched Images</th>
                    </tr>
                </thead>
                <tbody>
                    ${registrationResults.map(renderRegistrationResultRow).join("")}
                </tbody>
            </table>
        </div>
    `;
}

function renderRegistrationResultRow(result) {
    const statusClass = result.status === "success" ? "status-pill is-success" : "status-pill is-failed";
    const matchedImages = Array.isArray(result.matched_images) ? result.matched_images : [];

    return `
        <tr>
            <td>${escapeHtml(result.lpn)}</td>
            <td>${escapeHtml(result.route_from)}</td>
            <td>${escapeHtml(result.route_to)}</td>
            <td>${escapeHtml(result.flight)}</td>
            <td>${escapeHtml(result.date)}</td>
            <td>${escapeHtml(result.passenger)}</td>
            <td>${escapeHtml(result.baggage_class)}</td>
            <td>${escapeHtml(result.pieces)}</td>
            <td>${escapeHtml(result.baggage_type)}</td>
            <td><span class="${statusClass}">${escapeHtml(result.status)}</span></td>
            <td>${escapeHtml(result.method)}</td>
            <td>${renderMatchedImagesGallery(matchedImages)}</td>
        </tr>
    `;
}

function renderMatchedImagesGallery(images) {
    if (!images.length) {
        return '<span class="image-missing">No matched image</span>';
    }

    return `
        <div class="matched-image-gallery">
            ${images.map(renderMatchedImageCard).join("")}
        </div>
    `;
}

function renderMatchedImageCard(image) {
    const originalUrl = buildSessionImageUrl(image.original_image_path);
    const processedUrl = image.processed_image_path ? buildSessionImageUrl(image.processed_image_path) : null;
    const previewUrl = processedUrl || originalUrl;
    const label = image.photo_filename || "Matched image";
    const strategy = image.qr_strategy ? `<span class="image-meta">QR read: ${escapeHtml(image.qr_strategy)}</span>` : "";

    return `
        <div class="matched-image-card">
            ${renderImageTrigger(previewUrl, label)}
            ${strategy}
            ${processedUrl ? `<button class="text-link-button" type="button" data-image-url="${originalUrl}">Open Original</button>` : ""}
        </div>
    `;
}

function renderUnidentifiedImagesTable(unidentifiedImages) {
    if (!unidentifiedImages.length) {
        return '<div class="session-detail-loading">No unidentified images for this session.</div>';
    }

    return `
        <div class="session-detail-table-wrap">
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>Photo Filename</th>
                        <th>Original Image</th>
                        <th>Processed Image</th>
                    </tr>
                </thead>
                <tbody>
                    ${unidentifiedImages.map(renderUnidentifiedRow).join("")}
                </tbody>
            </table>
        </div>
    `;
}

function renderUnidentifiedRow(image) {
    const originalUrl = buildSessionImageUrl(image.original_image_path);
    const processedUrl = image.processed_image_path ? buildSessionImageUrl(image.processed_image_path) : null;

    return `
        <tr>
            <td>${escapeHtml(image.photo_filename)}</td>
            <td>${renderImageTrigger(originalUrl, image.photo_filename || "Original image")}</td>
            <td>${processedUrl ? renderImageTrigger(processedUrl, "Processed image") : '<span class="image-missing">-</span>'}</td>
        </tr>
    `;
}

function renderImageTrigger(url, label) {
    return `
        <button
            class="thumb-button"
            type="button"
            data-image-url="${url}"
            aria-label="Open ${escapeHtml(label)}"
        >
            <img class="thumb-image" src="${url}" alt="${escapeHtml(label)}">
            <span class="thumb-caption">Preview</span>
        </button>
    `;
}

function buildSessionImageUrl(path) {
    return `/static/sessions/${path}`;
}

function initializeImageModal() {
    const modal = document.getElementById("image-modal");
    const closeButton = document.getElementById("modal-close-button");

    if (!modal || modal.dataset.initialized === "true") {
        return;
    }

    modal.dataset.initialized = "true";
    modal.addEventListener("click", (event) => {
        if (event.target.dataset.closeModal === "true") {
            closeImageModal();
        }
    });
    closeButton.addEventListener("click", closeImageModal);
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeImageModal();
        }
    });
}

function openImageModal(url) {
    const modal = document.getElementById("image-modal");
    const image = document.getElementById("modal-image");
    image.src = url;
    modal.hidden = false;
    document.body.classList.add("modal-open");
}

function closeImageModal() {
    const modal = document.getElementById("image-modal");
    const image = document.getElementById("modal-image");
    if (!modal) {
        return;
    }
    modal.hidden = true;
    image.src = "";
    document.body.classList.remove("modal-open");
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
