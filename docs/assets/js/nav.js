/*
 * nav.js: builds the shared chrome (top bar, sidebar, prev/next, footer) of every tutorial page.
 * Each page sets <body data-page="05-training"> (the file name without .html, or "index").
 * Pages under tutorials/ set <body data-root=".."> so links resolve; index.html uses data-root=".".
 */
(function () {
  "use strict";
  const REPO = "https://github.com/Portiloop/portiloop-training";
  const TOC = [
    { group: "Getting started", items: [
      ["01-introduction", "Sleep spindles and closed-loop stimulation"],
      ["02-installation", "Installation and data"],
    ] },
    { group: "Understanding the pipeline", items: [
      ["03-preprocessing", "Signal preprocessing"],
      ["04-network", "Inside the network"],
    ] },
    { group: "Training", items: [
      ["05-training", "Training a new model"],
      ["06-evaluation", "Evaluating a detector"],
      ["07-hyperparameter-search", "Hyperparameter search"],
    ] },
    { group: "Using your model", items: [
      ["08-realtime", "Real-time simulation and stimulation"],
      ["09-your-data", "Detecting spindles in your own data"],
      ["10-deployment", "Exporting and deploying"],
    ] },
  ];
  window.PL_TOC = TOC;

  const body = document.body;
  const page = body.dataset.page || "index";
  const root = body.dataset.root || (page === "index" ? "." : "..");
  const href = (id) => (id === "index" ? root + "/index.html" : root + "/tutorials/" + id + ".html");
  const flat = [];
  TOC.forEach((g) => g.items.forEach((it) => flat.push(it)));

  // theme (stored per viewer, purely a convenience)
  const THEME_KEY = "pl-theme";
  try { const t = localStorage.getItem(THEME_KEY); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage unavailable */ }
  function toggleTheme() {
    const cur = document.documentElement.dataset.theme ||
      (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
    document.dispatchEvent(new Event("pl-theme-change"));
  }


  const top = document.createElement("header");
  top.className = "topbar";
  top.innerHTML =
    '<button class="menu-toggle" aria-label="Open navigation">☰</button>' +
    '<a class="brand" href="' + href("index") + '"><span class="logo-word" role="img" aria-label="Portiloop"></span><span class="brand-sub">Tutorials</span></a>' +
    '<span class="spacer"></span>' +
    '<button class="theme-toggle" aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>' +
    '<a class="icon-link" href="' + REPO + '" rel="noopener">GitHub</a>';
  body.prepend(top);
  top.querySelector(".theme-toggle").addEventListener("click", toggleTheme);
  top.querySelector(".menu-toggle").addEventListener("click", () => body.classList.toggle("nav-open"));

  const main = document.querySelector("main");
  const layout = document.createElement("div");
  layout.className = "layout";
  const side = document.createElement("nav");
  side.className = "sidebar";
  side.setAttribute("aria-label", "Tutorials");
  let html = '<ol><li><a href="' + href("index") + '"' + (page === "index" ? ' aria-current="page"' : "") + '><span class="num">⌂</span><span>Overview</span></a></li></ol>';
  TOC.forEach((g) => {
    html += "<h4>" + g.group + "</h4><ol>";
    g.items.forEach(([id, title]) => {
      const n = id.slice(0, 2);
      html += '<li><a href="' + href(id) + '"' + (id === page ? ' aria-current="page"' : "") + '><span class="num">' + n + "</span><span>" + title + "</span></a></li>";
    });
    html += "</ol>";
  });
  html += '<h4>Resources</h4><ol>' +
    '<li><a href="' + REPO + '"><span class="num">↗</span><span>Training code (GitHub)</span></a></li>' +
    '<li><a href="https://doi.org/10.1371/journal.pone.0270696"><span class="num">↗</span><span>Portiloop paper (PLOS ONE)</span></a></li>' +
    "</ol>";
  side.innerHTML = html;
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(side);
  layout.appendChild(main);
  main.addEventListener("click", () => body.classList.remove("nav-open"));

  // heading anchors
  main.querySelectorAll("h2[id], h3[id]").forEach((h) => {
    const a = document.createElement("a");
    a.className = "anchor"; a.href = "#" + h.id; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  // copy buttons on code blocks
  main.querySelectorAll("pre").forEach((pre) => {
    const b = document.createElement("button");
    b.className = "copy-btn"; b.type = "button"; b.textContent = "Copy";
    b.addEventListener("click", () => {
      const code = pre.querySelector("code") || pre;
      const done = () => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200); };
      if (navigator.clipboard) navigator.clipboard.writeText(code.innerText).then(done, () => {});
    });
    pre.appendChild(b);
  });

  // prev / next + footer
  const content = main.querySelector(".content") || main;
  const i = flat.findIndex(([id]) => id === page);
  if (i >= 0) {
    const pager = document.createElement("nav");
    pager.className = "pager";
    const prev = i > 0 ? flat[i - 1] : ["index", "Overview"];
    pager.innerHTML = '<a class="prev" href="' + href(prev[0]) + '"><div class="dir">← Previous</div>' + prev[1] + "</a>" +
      (i < flat.length - 1 ? '<a class="next" href="' + href(flat[i + 1][0]) + '"><div class="dir">Next →</div>' + flat[i + 1][1] + "</a>" : "");
    content.appendChild(pager);
  }
  const foot = document.createElement("footer");
  foot.className = "site-footer";
  const src = page === "index" ? "docs/index.html" : "docs/tutorials/" + page + ".html";
  foot.innerHTML = 'Portiloop training tutorials · MIT License · ' +
    '<a href="' + REPO + "/blob/main/" + src + '">Edit this page on GitHub</a> · ' +
    '<a href="' + REPO + '/issues">Report a problem</a>';
  content.appendChild(foot);

  // KaTeX auto-render if the page loaded it
  window.addEventListener("load", () => {
    if (window.renderMathInElement) {
      window.renderMathInElement(main, {
        delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
  });
})();
