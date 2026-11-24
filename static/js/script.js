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
      identifyBtn.disabled = !selectedFile;
    }
  });

  /* ---------- rendering (textContent only, so server text can never inject HTML) ---------- */
  function renderResult(data) {
    loading.hidden = true;
    $("resultImg").src = previewUrl;
    $("animalName").textContent = data.identified ? data.predicted_animal : "Not sure";
    $("confidenceValue").textContent = data.confidence.toFixed(1);

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
      badge.textContent = custom ? "Custom trained model active" : "Pretrained ImageNet model active";

      let text = s.name + " · " + (custom ? "custom-trained" : "pretrained on ImageNet") +
                 " · " + s.num_classes + (custom ? " classes" : " animal groups");
      if (custom && s.validation_accuracy != null) text += " · validation accuracy " + s.validation_accuracy + "%";
      if (custom && s.test_accuracy != null) text += " · test accuracy " + s.test_accuracy + "%";
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

  loadStatus();
})();
