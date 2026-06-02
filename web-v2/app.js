/* ============================================================
   SkinAI — shared behavior
   ============================================================ */
(function () {
  "use strict";

  /* ---- mark ready so gated entrance animations can play (base state stays visible) ---- */
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.add("is-ready")));

  /* ---- nav scroll state ---- */
  const nav = document.querySelector(".nav");
  if (nav) {
    const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* ---- reveal on scroll ---- */
  const reveals = document.querySelectorAll(".reveal");
  if (reveals.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    reveals.forEach((el) => io.observe(el));
  }

  /* ============================================================
     Analysis engine — shared by home demo + scan page
     ============================================================ */
  // Classes: abbr, full name, high-risk flag
  const CLASSES = [
    ["NV",   "Melanocytic nevus",        false],
    ["MEL",  "Melanoma",                 true],
    ["BCC",  "Basal cell carcinoma",     true],
    ["BKL",  "Benign keratosis",         false],
    ["AK",   "Actinic keratosis",        true],
    ["SCC",  "Squamous cell carcinoma",  true],
    ["DF",   "Dermatofibroma",           false],
    ["VASC", "Vascular lesion",          false],
    ["UNK",  "Unknown",                  false],
  ];

  const STEPS = [
    "Normalizing image · ImageNet stats",
    "EfficientNet-B4 · forward pass",
    "EfficientNet-B5 · forward pass",
    "EfficientNet-B7 · forward pass",
    "Soft-voting ensemble · averaging",
  ];

  // Deterministic-feeling but varied benign-leaning distribution.
  // Returns sorted [{abbr,name,high,p}], p as fraction summing ~1.
  window.SkinAI = window.SkinAI || {};

  window.SkinAI.classes = CLASSES;
  window.SkinAI.steps = STEPS;

  window.SkinAI.synthDistribution = function () {
    // Pick a "winner" — weighted toward benign nevus/keratosis for a reassuring demo,
    // with a real chance of a flagged result so the high-risk UI is demonstrable.
    const buckets = [
      { i: 0, w: 42 }, // NV
      { i: 3, w: 20 }, // BKL
      { i: 1, w: 14 }, // MEL
      { i: 2, w: 10 }, // BCC
      { i: 6, w: 6 },  // DF
      { i: 7, w: 4 },  // VASC
      { i: 4, w: 2 },  // AK
      { i: 5, w: 2 },  // SCC
    ];
    const total = buckets.reduce((s, b) => s + b.w, 0);
    let r = Math.random() * total, win = buckets[0].i;
    for (const b of buckets) { r -= b.w; if (r <= 0) { win = b.i; break; } }

    const top = 0.58 + Math.random() * 0.34; // 58–92%
    const probs = CLASSES.map((_, i) => (i === win ? top : Math.random()));
    // zero out winner before normalizing remainder
    const remainder = 1 - top;
    const others = probs.map((v, i) => (i === win ? 0 : v));
    const osum = others.reduce((s, v) => s + v, 0) || 1;
    const final = probs.map((v, i) => (i === win ? top : (others[i] / osum) * remainder));

    return CLASSES
      .map((c, i) => ({ abbr: c[0], name: c[1], high: c[2], p: final[i] }))
      .sort((a, b) => b.p - a.p);
  };

  /* ---- runs the analyzing animation in a container, then calls cb(dist) ---- */
  window.SkinAI.runAnalysis = function (opts) {
    const { stepEl, progEl, onDone, duration = 2600 } = opts;
    const dist = window.SkinAI.synthDistribution();
    const t0 = performance.now();
    let stepIdx = -1, finished = false;
    function paint(e) {
      if (progEl) progEl.style.width = (e * 100) + "%";
      const idx = Math.min(STEPS.length - 1, Math.floor(e * STEPS.length));
      if (idx !== stepIdx && stepEl) { stepIdx = idx; stepEl.textContent = STEPS[idx]; }
    }
    function done() { if (finished) return; finished = true; paint(1); setTimeout(() => onDone(dist), 220); }
    function frame(now) {
      const e = Math.min(1, (now - t0) / duration);
      paint(e);
      if (e < 1) requestAnimationFrame(frame);
      else done();
    }
    requestAnimationFrame(frame);
    // safety: timers survive rAF throttling (e.g. backgrounded tab) so the flow always completes
    setTimeout(done, duration + 400);
  };

  /* ---- helpers ---- */
  window.SkinAI.fmtPct = (p) => (p * 100).toFixed(p >= 0.1 ? 0 : 1) + "%";

  window.SkinAI.readImage = function (file, cb) {
    if (!file || !file.type.startsWith("image/")) { cb(null); return; }
    const r = new FileReader();
    r.onload = () => cb(r.result);
    r.readAsDataURL(file);
  };

  /* ============================================================
     Real backend integration — Modal API
     ============================================================ */
  const API_URL = "https://sh4wn27--skinai-api.modal.run";
  const HIGH_RISK_CODES = new Set(["MEL", "BCC", "SCC", "AK"]);

  // Convert a File to a JPEG Blob via canvas (mirrors ScanWidget.tsx)
  window.SkinAI.toJpeg = function (file) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      var url = URL.createObjectURL(file);
      img.onload = function () {
        var canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext("2d").drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        canvas.toBlob(
          function (blob) { blob ? resolve(blob) : reject(new Error("canvas conversion failed")); },
          "image/jpeg",
          0.92
        );
      };
      img.onerror = function () { URL.revokeObjectURL(url); reject(new Error("could not load image")); };
      img.src = url;
    });
  };

  // Hit the real Modal /predict endpoint.
  // opts: { file, stepEl, progEl, onDone(dist), onError(msg) }
  window.SkinAI.runRealAnalysis = function (opts) {
    var file = opts.file, stepEl = opts.stepEl, progEl = opts.progEl;
    var onDone = opts.onDone, onError = opts.onError;
    var apiSteps = [
      "Converting image",
      "Uploading to ensemble",
      "EfficientNet-B4 · forward pass",
      "EfficientNet-B5 · forward pass",
      "EfficientNet-B7 · soft vote",
    ];
    var pct = 0, sidx = -1;

    // Animate progress bar while the API call runs
    var ticker = setInterval(function () {
      if (pct < 82) {
        pct += 4 + Math.random() * 7;
        if (progEl) progEl.style.width = Math.min(82, pct).toFixed(1) + "%";
        var idx = Math.min(apiSteps.length - 1, Math.floor(pct / 18));
        if (idx !== sidx && stepEl) { sidx = idx; stepEl.textContent = apiSteps[idx]; }
      }
    }, 380);

    window.SkinAI.toJpeg(file)
      .then(function (jpeg) {
        var fd = new FormData();
        fd.append("file", jpeg, "image.jpg");
        return fetch(API_URL + "/predict", { method: "POST", body: fd });
      })
      .then(function (res) {
        if (!res.ok) throw new Error("Server error (" + res.status + ")");
        return res.json();
      })
      .then(function (data) {
        clearInterval(ticker);
        if (progEl) progEl.style.width = "100%";
        if (stepEl) stepEl.textContent = "Complete";
        // Map backend shape → internal shape
        var dist = (data.predictions || []).map(function (p) {
          return { abbr: p.code, name: p.class, high: HIGH_RISK_CODES.has(p.code), p: p.confidence };
        }).sort(function (a, b) { return b.p - a.p; });
        setTimeout(function () { onDone(dist); }, 220);
      })
      .catch(function (err) {
        clearInterval(ticker);
        if (onError) onError(err && err.message ? err.message : "Something went wrong. Please try again.");
      });
  };
})();
