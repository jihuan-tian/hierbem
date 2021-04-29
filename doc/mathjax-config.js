MathJax.Hub.Config({
  jax: ["input/TeX", "output/HTML-CSS"],
  extensions: ["tex2jax.js", "TeX/AMSmath.js", "TeX/AMSsymbols.js", "TeX/noUndefined.js"],
  tex2jax: {
    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
    skipTags: ["script","noscript","style","textarea","pre","code"],
    processEscapes: false,
    processEnvironments: true,
    preview: "TeX"
  },
  TeX: {
    Macros: {
      intd: "\\,{\\rm d}",
      diff: "{\\rm d}",
      Diff: "{\\rm D}",
      pdiff: "\\partial",
      DD: ["\\frac{\\diff}{\\diff #2}\\left( #1 \\right)", 2],
      Dd: ["\\frac{\\diff #1}{\\diff #2}", 2],
      PD: ["\\frac{\\pdiff}{\\pdiff #2}\\left( #1 \\right)", 2],
      Pd: ["\\frac{\\pdiff #1}{\\pdiff #2}", 2],
      rme: "{\\rm e}",
      rmi: "{\\rm i}",
      rmj: "{\\rm j}",
      vect: ["\\boldsymbol{#1}", 1],
      dform: ["\\overset{\\rightharpoonup}{\\boldsymbol{#1}}", 1],
      cochain: ["\\overset{\\rightharpoonup}{#1}", 1],
      Abs: ["\\big\\lvert#1\\big\\rvert", 1],
      abs: ["\\lvert#1\\rvert", 1],
      Norm: ["\\big\\lVert#1\\big\\rVert", 1],
      norm: ["\\lVert#1\\rVert", 1],
      normvect: "\\vect{n}",
      ouset: ["\\overset{#3}{\\underset{#2}{#1}}", 3],
      cscript: ["\\;\\; #1", 1],
      suchthat: "\\textit{S.T.\\;}",
      prefstar: "\\ast",
      restrict: "\\big\\vert",
      sgn: "{\\rm sgn}",
      erf: "{\\rm erf}",
      Bd: "{\\rm Bd}",
      Int: "{\\rm Int}",
      rank: "{\\rm rank}",
      divergence: "{\\rm div}",
      grad: "{\\rm grad}",
      tr: "{\\rm tr}",
      span: "{\\rm span}"
    },
    extensions: ["AMScd.js"],
    equationNumbers: { autoNumber: "AMS" }
  },
  "HTML-CSS": {
     availableFonts: ["STIX","TeX"],
     preferredFont: "TeX",
     webFont: "TeX",
     imageFont: "TeX",
     showMathMenu: true
  },
  MMLorHTML: {
    prefer: {
      MSIE:    "MML",
      Firefox: "MML",
      Opera:   "HTML",
      other:   "HTML"
    }
  }
});
