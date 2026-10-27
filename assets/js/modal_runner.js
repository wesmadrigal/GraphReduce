(function () {
  function normalizeBaseUrl(url) {
    return (url || "").trim().replace(/\/+$/, "");
  }

  function resolvedApiBase(container) {
    const input = container.querySelector("[data-api-input]");
    const fromInput = input ? normalizeBaseUrl(input.value) : "";
    const fromStorage = normalizeBaseUrl(localStorage.getItem("graphreduceApiBase") || "");
    const fromDataset = normalizeBaseUrl(container.dataset.apiBase || "");
    const fallback = "http://127.0.0.1:8001";
    return fromInput || fromStorage || fromDataset || fallback;
  }

  function setApiBaseUi(container) {
    const input = container.querySelector("[data-api-input]");
    if (!input) {
      return;
    }
    const val = resolvedApiBase(container);
    input.value = val;
  }

  function initModalRunner() {
    const containers = document.querySelectorAll("[data-modal-runner]");
    containers.forEach(function (container) {
      if (container.dataset.runnerBound === "1") {
        return;
      }
      container.dataset.runnerBound = "1";
      const runBtn = container.querySelector("[data-run-btn]");
      const saveBtn = container.querySelector("[data-save-api-btn]");
      if (!runBtn) {
        return;
      }
      setApiBaseUi(container);
      if (saveBtn) {
        saveBtn.addEventListener("click", function () {
          const input = container.querySelector("[data-api-input]");
          if (!input) {
            return;
          }
          const val = normalizeBaseUrl(input.value);
          if (val) {
            localStorage.setItem("graphreduceApiBase", val);
          }
        });
      }
      runBtn.addEventListener("click", function () {
        startRun(container);
      });
    });
  }

  function appendLine(logEl, line) {
    logEl.textContent += line + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function startRun(container) {
    const baseUrl = resolvedApiBase(container);
    const example = container.dataset.example || "hello_world";

    const runBtn = container.querySelector("[data-run-btn]");
    const statusEl = container.querySelector("[data-status]");
    const logEl = container.querySelector("[data-log]");

    runBtn.disabled = true;
    logEl.textContent = "";
    statusEl.textContent = "Starting job...";

    let jobId = null;
    try {
      const resp = await fetch(baseUrl + "/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ example }),
      });
      if (!resp.ok) {
        throw new Error("Job start failed: HTTP " + resp.status);
      }
      const payload = await resp.json();
      jobId = payload.job_id;
      statusEl.textContent = "Job running on " + baseUrl + " : " + jobId;
    } catch (err) {
      statusEl.textContent = "Failed to start job";
      appendLine(logEl, String(err));
      runBtn.disabled = false;
      return;
    }

    const streamUrl = baseUrl + "/jobs/" + encodeURIComponent(jobId) + "/stream";
    const src = new EventSource(streamUrl);

    src.onmessage = function (evt) {
      appendLine(logEl, evt.data);
    };

    src.addEventListener("done", async function () {
      src.close();
      try {
        const jobResp = await fetch(baseUrl + "/jobs/" + encodeURIComponent(jobId));
        const job = await jobResp.json();
        statusEl.textContent = "Finished with status: " + job.status + " (code=" + job.return_code + ")";
      } catch (_err) {
        statusEl.textContent = "Finished";
      }
      runBtn.disabled = false;
    });

    src.onerror = function () {
      appendLine(logEl, "[stream disconnected]");
      statusEl.textContent = "Stream error. Check docs/api server.";
      src.close();
      runBtn.disabled = false;
    };
  }

  document.addEventListener("DOMContentLoaded", initModalRunner);
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(initModalRunner);
  }
})();
