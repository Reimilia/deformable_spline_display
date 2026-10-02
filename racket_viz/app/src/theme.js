/**
 * Shared color palette ("Blueprint": technical drafting-table, deep navy with
 * chalk-white ink) for every canvas 2D and WebGL color in the app. Centralized
 * so a future theme swap (see the Storybook mockup, or reverting to
 * Observatory) touches this one file plus index.html's :root variables,
 * rather than hunting hex literals across scene files.
 *
 * NOTE: index.html's <style> block defines the same palette as CSS custom
 * properties for chrome (backgrounds, borders, buttons, sliders) -- there's
 * no build-time bridge between CSS and JS here, so the two are kept in sync
 * by hand. If you change one, change the other.
 */
export const THEME = {
  bg: "#0b1e33",
  panelBg: "#0f2743",
  border: "#2a4d73",
  text: "#eaf6ff",
  muted: "#9db6d1",

  gridLine: "rgba(234, 246, 255, 0.05)",

  stable: "#5ec8e0",
  unstable: "#ff8a75",
  imax: "#f4b942",
  current: "#ffe27a",
  target: "#e0a85c",
  separatrix: "#d97a4d",
  trajectory: "#3a7ca8",
  referenceSphere: "#bcd9ec",

  racketTube: "#f4b942",
  racketFaceA: "#ff8a75",
  racketFaceB: "#5ec8e0",
};
