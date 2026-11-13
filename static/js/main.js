// Interface : choix de l'image, envoi au serveur Flask, affichage du résultat.
const MAX_SIZE = 5 * 1024 * 1024; // 5 Mo, comme côté serveur
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");

const placeholder = document.getElementById("placeholder");
const errorBox = document.getElementById("error");
const resultContent = document.getElementById("result-content");
const verdictEmoji = document.getElementById("verdict-emoji");
const verdictTitle = document.getElementById("verdict-title");
const verdictNote = document.getElementById("verdict-note");
const bars = document.getElementById("bars");

let selectedFile = null;

function showError(message) {
  placeholder.hidden = true;
  resultContent.hidden = true;
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearResult() {
  errorBox.hidden = true;
  resultContent.hidden = true;
  placeholder.hidden = false;
}

function setFile(file) {
  clearResult();
  if (!file) return;
  if (!ALLOWED_TYPES.includes(file.type)) {
    showError("Format non pris en charge. Utilisez JPG, PNG ou WEBP.");
    return;
  }
  if (file.size > MAX_SIZE) {
    showError("Image trop lourde (maximum 5 Mo).");
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropzoneText.hidden = true;
  predictBtn.disabled = false;
  resetBtn.hidden = false;
}

function reset() {
  selectedFile = null;
  fileInput.value = "";
  preview.hidden = true;
  preview.removeAttribute("src");
  dropzoneText.hidden = false;
  predictBtn.disabled = true;
  resetBtn.hidden = true;
  clearResult();
}

function renderResult(data) {
  errorBox.hidden = true;
  placeholder.hidden = true;

  const { best, is_confident: isConfident, results } = data;
  verdictEmoji.textContent = isConfident ? best.emoji : "🤔";
  verdictTitle.textContent = isConfident ? best.name : "Je ne suis pas sûr";
  verdictNote.classList.toggle("is-unsure", !isConfident);
  verdictNote.textContent = isConfident
    ? `Certitude : ${best.probability} %`
    : `Meilleure hypothèse : ${best.name} (${best.probability} %). Essayez une photo plus nette ou un autre animal parmi chat, chien, oiseau.`;

  bars.innerHTML = "";
  results.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "bar" + (index === 0 ? " is-best" : "");

    const head = document.createElement("div");
    head.className = "bar-head";
    const name = document.createElement("span");
    name.textContent = `${item.emoji} ${item.name}`;
    const value = document.createElement("span");
    value.textContent = `${item.probability} %`;
    head.append(name, value);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.append(fill);

    li.append(head, track);
    bars.append(li);
    // Petit délai pour que la transition de largeur se déclenche
    requestAnimationFrame(() => { fill.style.width = `${item.probability}%`; });
  });

  resultContent.hidden = false;
}

async function predict() {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyse en cours…";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Une erreur est survenue.");
    } else {
      renderResult(data);
    }
  } catch (err) {
    showError("Impossible de contacter le serveur. Vérifiez qu’il est bien lancé.");
  } finally {
    predictBtn.textContent = "Identifier l’animal";
    predictBtn.disabled = selectedFile === null;
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
predictBtn.addEventListener("click", predict);
resetBtn.addEventListener("click", reset);

// Glisser-déposer
["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

// Accessibilité clavier : Entrée / Espace ouvrent le sélecteur de fichier
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
