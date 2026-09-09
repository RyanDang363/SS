const state = {
  videoId: "",
  jobId: "",
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function setText(id, value) {
  $(id).textContent = value;
}

function setBusy(isBusy) {
  for (const button of document.querySelectorAll("button")) {
    button.disabled = isBusy;
  }
}

function apiArtifactUrl(path) {
  return `/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${String(h).padStart(2, "0")}:${mm}:${ss}` : `${mm}:${ss}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

async function checkHealth() {
  try {
    await fetchJson("/health");
    setText("api-status", "API online");
  } catch (error) {
    setText("api-status", "API offline");
    $("api-status").classList.add("is-error");
  }
}

async function refreshVideo() {
  if (!state.videoId) return;
  const payload = await fetchJson(`/videos/${encodeURIComponent(state.videoId)}`);
  setText("video-chip", payload.video.video_id);
  setText("artifact-status", JSON.stringify(payload.artifacts, null, 2));
}

async function pollJob() {
  if (!state.jobId) return;
  const job = await fetchJson(`/jobs/${encodeURIComponent(state.jobId)}`);
  setText("job-status", job.status);
  setText("job-stage", job.stage);
  if (job.status === "failed") {
    setText("artifact-status", job.error || "Indexing failed.");
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  if (job.status === "succeeded") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    await refreshVideo();
  }
}

function renderEvidence(results) {
  const list = $("evidence-list");
  list.innerHTML = "";
  setText("evidence-count", `${results.length} result${results.length === 1 ? "" : "s"}`);
  for (const result of results) {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const text = result.transcript_text || result.ocr_text || result.vlm_caption || result.combined_text || "";
    const frames = (result.frame_paths || []).slice(0, 4).map((path) => (
      `<img src="${apiArtifactUrl(path)}" alt="Frame evidence at ${formatTime(result.start_time)}">`
    )).join("");
    item.innerHTML = `
      <div class="evidence-meta">
        <span>${formatTime(result.start_time)}-${formatTime(result.end_time)}</span>
        <span>${result.chunk_id}</span>
      </div>
      <div class="evidence-text">${escapeHtml(text)}</div>
      <div class="frames">${frames}</div>
    `;
    list.appendChild(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

$("video-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  setText("file-name", file ? file.name : "No file selected");
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  try {
    const form = new FormData(event.currentTarget);
    for (const key of ["title", "video_id"]) {
      const value = String(form.get(key) || "").trim();
      if (value) {
        form.set(key, value);
      } else {
        form.delete(key);
      }
    }
    const payload = await fetchJson("/videos", { method: "POST", body: form });
    state.videoId = payload.video_id;
    $("video-id").value = payload.video_id;
    await refreshVideo();
  } catch (error) {
    setText("artifact-status", error.message);
  } finally {
    setBusy(false);
  }
});

$("refresh-video").addEventListener("click", async () => {
  state.videoId = $("video-id").value.trim() || state.videoId;
  try {
    await refreshVideo();
  } catch (error) {
    setText("artifact-status", error.message);
  }
});

$("start-index").addEventListener("click", async () => {
  state.videoId = $("video-id").value.trim() || state.videoId;
  if (!state.videoId) return;
  setBusy(true);
  try {
    const payload = await fetchJson(`/videos/${encodeURIComponent(state.videoId)}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcription_provider: $("transcription-provider").value,
        embedding_provider: $("embedding-provider").value,
        chunk_seconds: Number($("chunk-seconds").value),
        skip_ocr: $("skip-ocr").checked,
        skip_captions: $("skip-captions").checked,
      }),
    });
    state.jobId = payload.job_id;
    setText("job-status", payload.status);
    setText("job-stage", payload.stage);
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollJob, 1500);
    await pollJob();
  } catch (error) {
    setText("job-status", "failed");
    setText("job-stage", error.message);
  } finally {
    setBusy(false);
  }
});

$("ask-button").addEventListener("click", async () => {
  state.videoId = $("video-id").value.trim() || state.videoId;
  const question = $("question").value.trim();
  if (!state.videoId || !question) return;
  setBusy(true);
  try {
    const base = `/videos/${encodeURIComponent(state.videoId)}`;
    const topK = Number($("top-k").value);
    const chunkSeconds = Number($("chunk-seconds").value);
    if ($("ask-mode").value === "search") {
      const results = await fetchJson(`${base}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: topK,
          chunk_seconds: chunkSeconds,
          provider: $("retrieval-provider").value,
        }),
      });
      setText("answer-output", JSON.stringify(results, null, 2));
      renderEvidence(results);
      return;
    }
    const answer = await fetchJson(`${base}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: topK,
        chunk_seconds: chunkSeconds,
        retrieval_provider: $("retrieval-provider").value,
        answer_provider: $("answer-provider").value,
      }),
    });
    setText("answer-output", answer.answer);
    renderEvidence(answer.retrieval_results || []);
  } catch (error) {
    setText("answer-output", error.message);
  } finally {
    setBusy(false);
  }
});

checkHealth();
