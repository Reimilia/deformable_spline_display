# Darboux Shape Lab — Interactive Deformable Spline Demo

A browser-based interactive visualization of shape deformation using cubic B-splines and optimal transport theory.

**Live Demo:** https://reimilia.github.io/deformable_spline_display/

## Features

- 🎨 **Interactive Control**: Drag spline control points in real-time
- ⚖️ **Multiple Potentials**: Sinkhorn OT, Sliced Wasserstein, Chamfer, KL/KDE, Rényi
- 📊 **Live Metrics**: Track mismatch, kinetic energy, self-intersections, and iterations
- ⚙️ **Full Control**: Adjust damping, time step, smoothness, and target geometry
- 📱 **Responsive Design**: Works on desktop and mobile

## Getting Started

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Reimilia/deformable_spline_display.git
   cd deformable_spline_display
   ```

2. Open `docs/index.html` in your browser

### Local Development

Serve the site locally:
```bash
cd docs
python -m http.server 8000
# or: npx http-server
```

Visit `http://localhost:8000`

## Publishing

See [PUBLISH.md](PUBLISH.md) for detailed publishing workflow.

**TL;DR:**
```bash
git add .
git commit -m "Update content"
git push origin main
```

GitHub Actions automatically publishes to GitHub Pages.

## Project Structure

```
deformable_spline_display/
├── docs/                           # 📤 Published site
│   ├── index.html                  # Main interactive demo
│   ├── summary.html                # 10-minute guided walkthrough
│   ├── memo.pdf                    # Research writeup
│   └── assets/                     # Supporting files
├── darboux-igph-pht-learning/      # Research code (optional)
├── .github/workflows/              # CI/CD automation
│   └── publish.yml                 # GitHub Pages deployment
├── PUBLISH.md                      # Publishing guide
└── README.md                       # This file
```

## About

This project demonstrates:
- Cubic B-spline representation and evaluation
- Optimal transport approximations for shape matching
- Structured differential equations for shape evolution
- Interactive browser-based scientific computing

## License

See LICENSE file in repository.

## Contact

Questions or feedback? Open an issue in the repository.
