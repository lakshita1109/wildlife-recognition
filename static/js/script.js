/* Wildlife AI – frontend logic. Talks to the Flask backend:
     GET  /api/status   -> which model is active
     POST /predict      -> classify the uploaded image (multipart field "image")  */

(function () {
  "use strict";

  const ALLOWED = ["jpg", "jpeg", "png", "webp", "bmp"];
  let maxMb = 8;
  let selectedFile = null;
  let previewUrl = null;
  let busy = false;

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone"), fileInput = $("fileInput"), browseBtn = $("browseBtn");
  const previewImg = $("previewImg"), dzEmpty = $("dzEmpty"), fileName = $("fileName");
  const identifyBtn = $("identifyBtn"), clearBtn = $("clearBtn"), alertBox = $("alertBox");
  const placeholder = $("resultPlaceholder"), loading = $("loading"), resultContent = $("resultContent");
  const infoCard = $("infoCard");

  /* ---------- helpers ---------- */
  function showAlert(message, kind) {
    alertBox.textContent = message;
    alertBox.className = "alert" + (kind === "info" ? " info" : "");
    alertBox.hidden = false;
  }
  function hideAlert() { alertBox.hidden = true; }

  function resetResults() {
    dropzone.classList.remove("locked");
    $("lockTag").hidden = true;
    resultContent.hidden = true;
    loading.hidden = true;
    infoCard.hidden = true;
    placeholder.hidden = false;
  }

  function validateFile(file) {
    const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!ALLOWED.includes(ext)) {
      return "Unsupported file type. Please upload a " + ALLOWED.map((e) => e.toUpperCase()).join(", ") + " image.";
    }
    if (file.type && !file.type.startsWith("image/")) {
      return "That file is not an image.";
    }
    if (file.size === 0) return "That file is empty.";
    if (file.size > maxMb * 1024 * 1024) {
      return "The image is too large. The maximum size is " + maxMb + " MB.";
    }
    return null;
  }

  function setFile(file) {
    hideAlert();
    resetResults();
    const problem = validateFile(file);
    if (problem) {
      showAlert(problem);
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    previewImg.hidden = false;
    dzEmpty.hidden = true;
    fileName.textContent = file.name;
    identifyBtn.disabled = false;
    clearBtn.hidden = false;
  }

  function clearFile() {
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewImg.hidden = true;
    previewImg.removeAttribute("src");
    dzEmpty.hidden = false;
    fileName.textContent = "";
    fileInput.value = "";
    identifyBtn.disabled = true;
    clearBtn.hidden = true;
    hideAlert();
    resetResults();
  }

  /* ---------- upload interactions ---------- */
  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener("click", () => { if (!selectedFile) fileInput.click(); });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    if (files.length > 1) showAlert("Please drop one image at a time. Using the first file.", "info");
    setFile(files[0]);
  });
  // Stop the browser from opening a file dropped outside the dropzone
  ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));
  clearBtn.addEventListener("click", clearFile);

  /* ---------- identify ---------- */
  identifyBtn.addEventListener("click", async () => {
    if (!selectedFile) { showAlert("No image selected. Please choose an image first."); return; }
    if (busy) return;
    busy = true;
    hideAlert();
    identifyBtn.disabled = true;
    placeholder.hidden = true;
    resultContent.hidden = true;
    infoCard.hidden = true;
    loading.hidden = false;
    dropzone.classList.add("scanning");
    dropzone.classList.remove("locked");
    $("lockTag").hidden = true;
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });

    const form = new FormData();
    form.append("image", selectedFile);

    try {
      const response = await fetch("/predict", { method: "POST", body: form });
      let data = null;
      try { data = await response.json(); } catch (_) { /* non-JSON error page */ }

      if (!response.ok || !data || !data.success) {
        const msg = (data && data.error) || "The server returned an error (" + response.status + ").";
        loading.hidden = true;
        placeholder.hidden = false;
        showAlert(msg);
      } else {
        renderResult(data);
      }
    } catch (err) {
      loading.hidden = true;
      placeholder.hidden = false;
      showAlert("Cannot reach the Wildlife AI server. Make sure 'python app.py' is still running, then try again.");
    } finally {
      busy = false;
      dropzone.classList.remove("scanning");
      identifyBtn.disabled = !selectedFile;
    }
  });

  /* ---------- rendering (textContent only, so server text can never inject HTML) ---------- */
  function renderResult(data) {
    loading.hidden = true;
    $("resultImg").src = previewUrl;
    $("animalName").textContent = data.identified ? data.predicted_animal : "Not sure";
    showConfidence(data.confidence, data.identified);
    lockOn(data);

    const msg = $("resultMessage");
    if (data.message) { msg.textContent = data.message; msg.hidden = false; } else { msg.hidden = true; }

    const list = $("barsList");
    list.textContent = "";
    data.top_predictions.forEach((p) => {
      const li = document.createElement("li");
      const row = document.createElement("div");
      row.className = "bar-label";
      const name = document.createElement("span");
      name.textContent = p.animal;
      const pct = document.createElement("span");
      pct.textContent = p.confidence.toFixed(1) + "%";
      row.append(name, pct);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      track.appendChild(fill);
      li.append(row, track);
      list.appendChild(li);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = Math.max(p.confidence, 1) + "%"; }));
    });
    resultContent.hidden = false;

    if (data.animal_info) renderInfo(data.animal_info);
  }

  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const RING_LENGTH = 326.7;

  function showConfidence(value, identified) {
    const ring = $("ringFill"), num = $("confidenceValue");
    ring.classList.toggle("low", !identified);
    ring.style.strokeDashoffset = RING_LENGTH;
    if (REDUCED) {
      num.textContent = value.toFixed(1);
      ring.style.strokeDashoffset = RING_LENGTH * (1 - value / 100);
      return;
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ring.style.strokeDashoffset = RING_LENGTH * (1 - value / 100);
    }));
    const start = performance.now(), duration = 1300;
    (function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      num.textContent = (value * (1 - Math.pow(1 - t, 3))).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
    })(start);
  }

  function lockOn(data) {
    const tag = $("lockTag");
    tag.textContent = (data.identified ? data.predicted_animal : "Not sure") + "  " + data.confidence.toFixed(1) + "%";
    tag.className = "lock-tag" + (data.identified ? "" : " unsure");
    tag.hidden = false;
    dropzone.classList.add("locked");
    if (data.identified) leafBurst();
  }

  function leafBurst() {
    if (REDUCED) return;
    const ring = document.querySelector(".ring");
    const colors = ["#a8d5a2", "#2d5a3d", "#8b6b4a", "#6fa876"];
    for (let i = 0; i < 18; i++) {
      const leaf = document.createElement("span");
      const angle = (i / 18) * Math.PI * 2 + Math.random() * 0.4;
      const dist = 70 + Math.random() * 60;
      leaf.className = "leaf";
      leaf.style.setProperty("--dx", (Math.cos(angle) * dist).toFixed(0) + "px");
      leaf.style.setProperty("--dy", (Math.sin(angle) * dist).toFixed(0) + "px");
      leaf.style.setProperty("--rot", (Math.random() * 540 - 270).toFixed(0) + "deg");
      leaf.style.setProperty("--c", colors[i % colors.length]);
      ring.appendChild(leaf);
      setTimeout(() => leaf.remove(), 1500);
    }
  }

  function buildMarquee(names) {
    const box = $("marquee"), track = $("marqueeTrack");
    if (!names || !names.length) return;
    let items = names.slice();
    while (items.length < 24) items = items.concat(names);
    track.textContent = "";
    for (let copy = 0; copy < 2; copy++) {          // two identical halves = seamless loop
      items.forEach((name) => {
        const el = document.createElement("span");
        el.className = "marquee-item";
        el.textContent = name;
        track.appendChild(el);
      });
    }
    track.style.animationDuration = Math.max(40, items.length * 2.2) + "s";
    box.hidden = false;
  }

  function renderInfo(info) {
    $("infoName").textContent = info.name;
    $("infoCategory").textContent = info.category || "–";
    $("infoHabitat").textContent = info.habitat || "–";
    $("infoDiet").textContent = info.diet || "–";
    const facts = $("infoFacts");
    facts.textContent = "";
    (info.facts || []).forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      facts.appendChild(li);
    });
    infoCard.hidden = false;
  }

  /* ---------- model status (badge + About section) ---------- */
  async function loadStatus() {
    const badge = $("modeBadge"), details = $("modelDetails");
    try {
      const res = await fetch("/api/status");
      const s = await res.json();
      maxMb = s.max_upload_mb || maxMb;

      if (!s.ready) {
        badge.textContent = "Model unavailable";
        details.textContent = s.error || "The AI model is not loaded.";
        showAlert(s.error || "The AI model is not loaded.");
        return;
      }
      const custom = s.mode === "custom";
      const hybrid = s.mode === "hybrid";
      badge.textContent = custom ? "Custom trained model active"
        : hybrid ? "Hybrid model active (animals + pet breeds)" : "Pretrained ImageNet model active";

      let text = s.name + " · " + (custom ? "custom-trained" : hybrid ? "pretrained animals + custom-trained breeds" : "pretrained on ImageNet") +
                 " · " + s.num_classes + (custom ? " classes" : " labels");
      if ((custom || hybrid) && s.validation_accuracy != null) text += " · custom validation accuracy " + s.validation_accuracy + "%";
      if ((custom || hybrid) && s.test_accuracy != null) text += " · custom test accuracy " + s.test_accuracy + "%";
      details.textContent = text;
      if (s.notice) showAlert(s.notice, "info");

      if (s.supported_animals && s.supported_animals.length) {
        $("supportedCount").textContent = s.supported_animals.length;
        const chips = $("supportedList");
        chips.textContent = "";
        s.supported_animals.forEach((a) => {
          const c = document.createElement("span");
          c.className = "chip";
          c.textContent = a;
          chips.appendChild(c);
        });
        $("supportedWrap").hidden = false;
        buildMarquee(s.supported_animals);
      }
    } catch (_) {
      badge.textContent = "Server unreachable";
      details.textContent = "Cannot reach the server.";
      showAlert("Cannot reach the Wildlife AI server. Make sure 'python app.py' is running.");
    }
  }

  /* ---------- mobile nav ---------- */
  const navToggle = $("navToggle"), navLinks = $("navLinks");
  navToggle.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  navLinks.addEventListener("click", () => {
    navLinks.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  });

  /* ---------- navbar: solid on scroll, sliding pill on the active section ---------- */
  const navbar = $("navbar"), pill = $("navPill");
  const links = Array.from(navLinks.querySelectorAll("a[data-section]"));

  function movePill(link) {
    links.forEach((l) => l.classList.toggle("active", l === link));
    if (!link || getComputedStyle(pill).display === "none") return;
    pill.style.width = link.offsetWidth + "px";
    pill.style.transform = "translateX(" + link.offsetLeft + "px)";
  }
  function currentSection() {
    let found = links[0];
    links.forEach((l) => {
      const el = document.getElementById(l.dataset.section);
      if (el && el.getBoundingClientRect().top <= window.innerHeight * 0.4) found = l;
    });
    return found;
  }

  /* ---------- hero parallax: scroll + pointer, and the night-vision scope ---------- */
  const layers = Array.from(document.querySelectorAll(".layer"));
  const hero = $("home");
  let ticking = false, px = 0;

  function applyLayers() {
    if (REDUCED) return;
    const y = window.scrollY;
    if (y > window.innerHeight * 1.2) return;
    layers.forEach((layer) => {
      const sp = parseFloat(layer.dataset.speed);
      layer.style.transform = "translate3d(" + (px * sp * -60).toFixed(1) + "px," + (y * sp).toFixed(1) + "px,0)";
    });
  }
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      navbar.classList.toggle("scrolled", window.scrollY > 40);
      movePill(currentSection());
      applyLayers();
      ticking = false;
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();

  if (!REDUCED && window.matchMedia("(pointer: fine)").matches) {
    let raf = 0;
    hero.addEventListener("pointerenter", () => hero.classList.add("hunting"));
    hero.addEventListener("pointerleave", () => { hero.classList.remove("hunting"); px = 0; applyLayers(); });
    hero.addEventListener("pointermove", (e) => {
      const r = hero.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        hero.style.setProperty("--mx", (x * 100).toFixed(1) + "%");
        hero.style.setProperty("--my", (y * 100).toFixed(1) + "%");
        px = x - 0.5;
        applyLayers();
      });
    });
  }

  /* ---------- fireflies ---------- */
  if (!REDUCED) {
    const box = $("fireflies");
    for (let i = 0; i < 16; i++) {
      const f = document.createElement("span");
      f.className = "firefly";
      f.style.left = (Math.random() * 100).toFixed(1) + "%";
      f.style.top = (30 + Math.random() * 55).toFixed(1) + "%";
      f.style.setProperty("--s", (3 + Math.random() * 3).toFixed(1) + "px");
      f.style.setProperty("--d", (7 + Math.random() * 7).toFixed(1) + "s");
      f.style.setProperty("--delay", (-Math.random() * 10).toFixed(1) + "s");
      f.style.setProperty("--x", ((Math.random() - 0.5) * 120).toFixed(0) + "px");
      f.style.setProperty("--y", ((Math.random() - 0.5) * 90).toFixed(0) + "px");
      box.appendChild(f);
    }
  }

  /* ---------- reveal About cards on scroll ---------- */
  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !REDUCED) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.15 });
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }

  loadStatus();
})();