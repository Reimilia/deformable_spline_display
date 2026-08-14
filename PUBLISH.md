# Publishing Guide

This repository is configured for automated publishing to GitHub Pages.

## How to Publish

### Automatic (Recommended)
Simply push to `main` branch, and GitHub Actions will automatically deploy the contents of the `docs/` folder to GitHub Pages:

```bash
git add .
git commit -m "Update demo content"
git push origin main
```

The workflow file at `.github/workflows/publish.yml` handles the rest.

### Manual Check
View deployment status at: **Settings → Pages** in your GitHub repository

## Project Structure

```
docs/              → Published site (synced to GitHub Pages)
  ├── index.html   → Main interactive demo
  ├── summary.html → Guided walkthrough
  ├── memo.pdf     → Research memo
  └── assets/      → CSS, JavaScript, images
```

## Updating Content

1. **To update the demo:** Edit files in `docs/`
2. **To add new pages:** Create `.html` files in `docs/`
3. **Push to deploy:** `git push origin main`

Changes appear live at: `https://username.github.io/deformable_spline_display/`

## Development Workflow

### Branch Strategy
- **main**: Source code (published automatically)
- **develop**: Development work (optional)

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, test locally
# Then merge back to main
git checkout main
git merge feature/your-feature
git push origin main
```

### Local Testing
If you want to test your site locally before publishing:

```bash
# Python 3
cd docs
python -m http.server 8000

# Or use Node.js http-server
npx http-server docs -p 8000
```

Then visit `http://localhost:8000`

## CI/CD Pipeline

The GitHub Actions workflow:
1. ✅ Triggers on push to `main`
2. ✅ Checks out code
3. ✅ Uploads `docs/` folder as artifact
4. ✅ Deploys to GitHub Pages

View logs at: **Actions** tab in your repository

## Key Files

- `.github/workflows/publish.yml` - Deployment automation
- `.gitignore` - Prevents node_modules, build artifacts, etc.
- `docs/` - Published content (this is what visitors see)

## Tips

- Keep large files out of this repo (use `.gitignore`)
- Test links before pushing (relative paths matter!)
- Use descriptive commit messages
- If changes don't appear, wait 1-2 minutes for GitHub Actions to complete
