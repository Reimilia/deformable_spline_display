# Publish the demo with GitHub Pages

The repository includes `.github/workflows/pages.yml`. It installs frontend
dependencies, validates the bundled preview data, runs the JavaScript tests,
builds both demo pages and publishes only `racket_viz/app/dist`. Pushes to your
repository's default branch publish automatically. You can also run the
workflow manually from the Actions tab.

## First deployment

1. Put the contents of this archive's `LieSPHGP-feature-racket-viz` folder at
   your GitHub repository root. Include `.github/workflows/pages.yml`,
   `racket_viz/app/package-lock.json`, and `racket_viz/data/previews/`.
2. Commit and push those files to the default branch. The root of the GitHub
   repository should contain `README.md`, `run_demo.py`, `.github/` and
   `racket_viz/`; avoid nesting the entire repository inside an extra folder.
3. In the repository's **Settings → Pages → Build and deployment**, set
   **Source** to **GitHub Actions**.
4. Under **Actions**, select **Publish deformation demo to GitHub Pages** and
   run it, or push a new commit to the default branch.
5. Open the URL shown by the `deploy` job or in Settings → Pages. A project
   repository normally uses `https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/`.

The workflow uses `configure-pages` to set Vite's base path automatically. It
supports project repositories, account Pages repositories and a configured
custom domain without hardcoding the repository name in JavaScript. If using
a custom domain, configure it in Settings → Pages before publishing.

## What works on the published page

GitHub Pages hosts static HTML, JavaScript and data files. The deformation page
therefore defaults to **Saved previews**:

- All **58 checkpoints** are selectable. Each has a saved feasible-test task
  and a hard OOD task, giving **116 real checkpoint rollouts**.
- Shape, phase space, energy/error charts, playback, scrubbing, frame/coordinate
  selection, reference/target-point overlays and JSON export work in the browser.
- Stage comparisons use the same original saved task and are available within
  a checkpoint variant. This preserves the exact comparison target and initial
  state. Live dynamics retains broader comparison choices within each family.
- Physics parameters are shown with the saved rollout values and are disabled
  in preview mode. Connect a dynamics service to recompute after changing them.
- The original racket demo still computes its dynamics directly in the browser
  and is accessible via **Racket demo ↗**.

The preview data come from the supplied Python integrators and checkpoints;
they are not surrogate JavaScript dynamics. For transfer checkpoints the saved
hard task still uses the original hard-test cache. Geometry is sampled at up
to 80 points per curve and values are serialized with eight significant digits
to keep the static site small. All native phase/metric timestamps are retained.

## Optional live dynamics from the Pages frontend

Host the Python service separately behind HTTPS. GitHub Pages serves the
frontend; the external service handles PyTorch inference. Start the service
with your Pages origin allowed, for example:

```bash
python -m pip install -r racket_viz/multilink_backend/requirements.txt
python run_demo.py --host 0.0.0.0 --port 8765 --allow-origin https://YOUR-USERNAME.github.io
```

The `--allow-origin` value is the page's origin, with no repository path and
no trailing slash. Repeat it for another allowed origin, such as a custom
domain. An HTTPS reverse proxy or your hosting platform should expose the
service's `/api/checkpoints` and `/api/rollout` endpoints.

To connect, open **Dynamics service** in the page's sidebar, enter the service
URL (for example `https://dynamics.example.org`) and press **Connect**. The
frontend handles JSON POST preflight requests and the service emits CORS
headers for the specified origin. Press **Saved previews** to return to
server-free playback.

To make the hosted service the initial default, add an Actions **repository
variable** called `DEMO_API_URL` containing its HTTPS URL, then rerun the Pages
workflow. Its value is included in the published frontend. If the service is unavailable at startup,
the frontend falls back to bundled previews. Each browser can override this
initial choice through the sidebar.

## Refresh previews after changing checkpoints

Ordinary deployments use the committed preview files, so Node alone is needed
for the default Pages build. After updating model files or wanting additional
saved tasks, generate previews locally from the repository root:

```bash
python -m pip install -r racket_viz/multilink_backend/requirements.txt
python racket_viz/multilink_backend/export_previews.py --tasks 1 --horizon 2
git add racket_viz/data/previews
git commit -m "Refresh checkpoint previews"
git push
```

Increase `--tasks` to include more feasible/hard task indices per checkpoint.
The exporter respects the size of each original saved task set. A different
`--horizon` is recorded in the preview manifest and displayed by the frontend.
It regenerates the manifest and removes stale JSON files only from its output
preview folder.

Alternatively, run the Pages workflow manually and check **refresh_previews**.
That run installs CPU PyTorch and regenerates the previews before publishing.
It does not commit the refreshed files to your repository; regenerate and
commit locally if future ordinary builds should reuse the updated data.

## Test the static build locally

After generating or updating previews, from `racket_viz/app`:

```bash
npm ci
npm run validate:previews
npm test
VITE_STATIC_DEMO=true VITE_BASE_PATH=/ npm run build
npm run preview
```

The last build command uses bash/WSL environment-variable syntax. In PowerShell
set `$env:VITE_STATIC_DEMO="true"` and `$env:VITE_BASE_PATH="/"` before running
`npm run build`. Ordinary `npm run build` retains local live-dynamics startup
and falls back to previews if the service is absent.

For a repository-path rehearsal, build with
`VITE_BASE_PATH=/YOUR-REPOSITORY/` and open the corresponding path served by
Vite's preview server.

Reference documentation:

- [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Vite static deployment](https://vite.dev/guide/static-deploy)
- [GitHub Pages overview](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
