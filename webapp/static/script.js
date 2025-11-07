const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadResult = document.getElementById("uploadResult");
const toast = document.getElementById("toast");

// helper
function showToast(text) {
    toast.textContent = text; toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 1600);
}

// Upload image
uploadBtn.onclick = async () => {
    const f = fileInput.files[0];
    if (!f) { showToast("Pehlay image select karo"); return; }
    const form = new FormData();
    form.append("file", f);
    uploadResult.textContent = "Predicting...";
    try {
        const res = await fetch("/predict", { method: "POST", body: form });
        const data = await res.json();
        if (data.error) { uploadResult.textContent = "Error: " + data.error; return; }
        const cls = data.label;
        const conf = data.confidence;
        const badge = document.createElement("span");
        badge.className = "badge " + (cls === "Mask" ? "mask" : "nomask");
        badge.textContent = cls;
        uploadResult.innerHTML = "";
        uploadResult.appendChild(badge);
        uploadResult.append(" Confidence: " + (conf * 100).toFixed(2) + "%");
        // small satisfying effect
        showToast("Done ✓");
    } catch (e) {
        uploadResult.textContent = "Error";
        console.error(e);
    }
};

// Webcam
const startBtn = document.getElementById("startCam");
const stopBtn = document.getElementById("stopCam");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const webcamResult = document.getElementById("webcamResult");
let stream = null;
let intervalId = null;

startBtn.onclick = async () => {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    // send frames every 600ms
    intervalId = setInterval(captureAndSend, 600);
};

stopBtn.onclick = () => {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    if (intervalId) clearInterval(intervalId);
};

async function captureAndSend() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/jpeg", 0.8);
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataURL })
        });
        const data = await res.json();
        if (data.error) { webcamResult.textContent = data.error; return; }
        // If faces details exist, show top face's result else global label
        let dispLabel = data.label;
        let conf = data.confidence;
        if (data.faces && data.faces.length > 0) {
            // pick first face
            dispLabel = data.faces[0].label;
            conf = data.faces[0].confidence;
        }
        webcamResult.innerHTML = `<span class="badge ${dispLabel === "Mask" ? "mask" : "nomask"}">${dispLabel}</span> ${(conf * 100).toFixed(1)}%`;
    } catch (e) {
        console.error(e);
    }
}
