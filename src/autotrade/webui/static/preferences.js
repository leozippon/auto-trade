/* The device's display preferences, applied before the first paint.

   Loaded as a classic, parser-blocking script ahead of the stylesheet, so the
   root element already carries its theme and zoom when anything is drawn: a
   dark-mode reader never sees a light frame, and a zoomed page never jumps
   size once app.js arrives. External rather than inline because the public
   site's CSP allows scripts from 'self' only. The theme is the stored choice,
   else the system's; app.js owns the toggle and the zoom selector. */
(function () {
  var root = document.documentElement;
  var theme = null;
  var zoom = null;
  try {
    theme = localStorage.getItem("ch_theme");
    zoom = localStorage.getItem("ch_zoom");
  } catch (error) {
    /* storage blocked: the system theme, no zoom */
  }
  if (theme !== "dark" && theme !== "light")
    theme =
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
  root.dataset.theme = theme;
  if (zoom && Number(zoom) > 0) root.style.setProperty("--ui-zoom", zoom);
})();
