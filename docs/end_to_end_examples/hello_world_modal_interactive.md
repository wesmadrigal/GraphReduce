# Hello World: Interactive Runner

This page runs the Hello World GraphReduce example on the same machine as the
docs API server and streams stdout live into the docs UI.

<div class="modal-runner" data-modal-runner data-api-base="http://127.0.0.1:8001">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="http://127.0.0.1:8001" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run Hello World</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>

## One-Time Local Setup

```bash
pip install graphreduce
pip install -r docs/api/requirements.txt
```

This Hello World runner uses the pandas backend, so optional extras like
`[spark]`, `[daft]`, or `[duckdb]` are not required.

## Start The Stream API (Terminal 1)

```bash
python docs/api/modal_stream_server.py
```

## Start Docs (Terminal 2)

```bash
mkdocs serve
```

Then open this page and click **Run Hello World**.

## Production Deployment (GitHub Pages + External Backend)

1. Deploy docs to GitHub Pages as usual.
2. Deploy the API server (`docs/api/modal_stream_server.py`) on your own host
   (DigitalOcean, Render, Fly, etc.) behind HTTPS.
3. Set backend CORS env var to your docs origin:

```bash
export GRAPHREDUCE_ALLOWED_ORIGINS="https://<your-gh-pages-or-custom-domain>"
```

4. In this page, set the backend URL in the input (for example
   `https://runner.yourdomain.com`) and click **Save API URL**.

## Notes

* Everything for this implementation lives under `docs/*`.
* The browser UI is static; command execution happens through `docs/api/modal_stream_server.py`.
* The API runs `python examples/hello_world_local_runner.py` locally on the server host.
* API base URL defaults to `http://127.0.0.1:8001` and can be overridden in the input.
