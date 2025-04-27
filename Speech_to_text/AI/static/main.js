let recorder = null;
let chunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const fileInput = document.getElementById("fileInput");
const plainInput = document.getElementById("plain");
const encryptedInput = document.getElementById("encrypted");
const decryptedInput = document.getElementById("decrypted");

recordBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunks = [];

    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" }); // Create file
      await upload(blob);                                    // 🔥 Auto upload ngay
    };

    recorder.start();
    recordBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (err) {
    alert("🎙️ Microphone error: " + err.message);
  }
};

stopBtn.onclick = () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    recordBtn.disabled = false;
    stopBtn.disabled = true;
  }
};

fileInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (file) {
    await upload(file);
  }
};

async function upload(file) {
  const form = new FormData();
  form.append("audio", file, "speech.wav");

  try {
    const response = await fetch("/infer", {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Server error: ${response.status}`);
    }

    const data = await response.json();
    plainInput.value = data.plain || "";
    encryptedInput.value = data.encrypted_base64 || "";
    decryptedInput.value = data.decrypted || "";

  } catch (error) {
    alert("❌ Upload error: " + error.message);
    console.error(error);
  }
}
