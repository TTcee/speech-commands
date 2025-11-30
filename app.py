# app.py
"""
МОДУЛЬ: app.py
=================
Призначення:
------------
- Піднімає Flask-сервіс для розпізнавання голосових команд
- Дає простий веб-інтерфейс:
    1) Завантажити WAV-файл
    2) Записати звук з мікрофона прямо в браузері (WAV)
- Використовує натреновану модель SpeechCommandCNN (yes/no/up/down)
"""

import os
import torch
import torchaudio
import soundfile as sf
from flask import Flask, request, jsonify, render_template_string
from model import SpeechCommandCNN
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# --- Константи ---
LABELS = ["yes", "no", "up", "down"]
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask-додаток ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Пристрій і модель ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    """
    Завантаження моделі з файлу saved_model/model.pth
    """
    model = SpeechCommandCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load("saved_model/model.pth", map_location=device))
    model.eval()
    return model


model = load_model(device)
print(f"✅ Модель завантажена ({device})")


# --- Преобробка аудіо ---
def preprocess_audio(waveform):
    """
    Перетворення сирого сигналу (16 kHz, 1 канал) у мел-спектрограму.
    На виході тензор форми [1, 64, time].
    """
    transform = torch.nn.Sequential(
        MelSpectrogram(sample_rate=16000, n_mels=64),
        AmplitudeToDB()
    )
    spec = transform(waveform)
    return spec


# --- HTML-інтерфейс з кнопками файлу і мікрофона ---
HTML_PAGE = """
<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Speech Commands Demo</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
    h1 { font-size: 24px; }
    .card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
    button { padding: 8px 16px; margin-top: 8px; cursor: pointer; }
    #result { font-weight: bold; margin-top: 16px; }
  </style>
  <!-- Підключаємо Recorder.js (робить WAV прямо в браузері) -->
  <script src="https://cdn.jsdelivr.net/gh/mattdiamond/Recorderjs@master/dist/recorder.js"></script>
</head>
<body>

<h1>Розпізнавання голосових команд (yes / no / up / down)</h1>

<div class="card">
  <h3>1. Завантажити WAV-файл</h3>
  <form id="uploadForm">
    <input type="file" name="file" accept=".wav" required>
    <br>
    <button type="submit">Відправити</button>
  </form>
</div>

<div class="card">
  <h3>2. Записати з мікрофона</h3>
  <p id="status">Натисни "Записати" і скажи команду.</p>
  <button id="recordBtn">Записати</button>
</div>

<h3 id="result">Результат: —</h3>

<script>
  const resultDiv = document.getElementById("result");
  const statusText = document.getElementById("status");
  const recordBtn = document.getElementById("recordBtn");

  // ----- Відправка готового WAV-файлу -----
  document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    resultDiv.textContent = "Обробка...";
    try {
      const resp = await fetch("/predict", { method: "POST", body: formData });
      const data = await resp.json();
      if (data.prediction) {
        resultDiv.textContent = "Результат: " + data.prediction;
      } else {
        resultDiv.textContent = "Помилка: " + (data.error || "невідома помилка");
      }
    } catch (err) {
      resultDiv.textContent = "Помилка запиту: " + err;
    }
  });

  // ----- Запис з мікрофона через Recorder.js (WAV) -----
  let audioContext;
  let gumStream;
  let rec; // Recorder.js інстанс
  let input;

  recordBtn.onclick = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();

      gumStream = stream;
      input = audioContext.createMediaStreamSource(stream);

      rec = new Recorder(input, { numChannels: 1 });

      statusText.textContent = "Запис триває... Скажи команду.";
      recordBtn.disabled = true;

      rec.record();

      // Авто-стоп через 1 секунду
      setTimeout(() => {
        rec.stop();
        statusText.textContent = "Обробка запису...";
        gumStream.getAudioTracks()[0].stop();

        rec.exportWAV(async (blob) => {
          const formData = new FormData();
          formData.append("file", blob, "mic_recording.wav");

          try {
            const resp = await fetch("/predict", { method: "POST", body: formData });
            const data = await resp.json();
            if (data.prediction) {
              resultDiv.textContent = "Результат: " + data.prediction;
            } else {
              resultDiv.textContent = "Помилка: " + (data.error || "невідома помилка");
            }
          } catch (err) {
            resultDiv.textContent = "Помилка запиту: " + err;
          } finally {
            recordBtn.disabled = false;
            statusText.textContent = "Натисни \\"Записати\\" і скажи команду.";
            rec.clear();
          }
        });
      }, 1000);

    } catch (err) {
      console.error(err);
      statusText.textContent = "Не вдалося отримати доступ до мікрофона.";
    }
  };
</script>

</body>
</html>
"""


# --- Роут головної сторінки ---
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


# --- Роут для прогнозу ---
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Файл не знайдено"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Ім'я файлу порожнє"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        # 📥 Читаємо аудіо БЕЗ torchaudio.load — через soundfile
        # waveform_np: [num_samples] або [num_samples, channels]
        waveform_np, sample_rate = sf.read(filepath, dtype="float32")

        import numpy as np
        if waveform_np.ndim == 1:
            # [num_samples] -> [1, num_samples]
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            # [num_samples, channels] -> моно -> [1, num_samples]
            mono = waveform_np.mean(axis=1)
            waveform = torch.from_numpy(mono).unsqueeze(0)

        # 🔥 Ресемплінг до 16 kHz, якщо треба
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Перетворення у спектрограму (тепер завжди 16 kHz)
        spec = preprocess_audio(waveform)       # [1, 64, time]
        spec = spec.unsqueeze(0).to(device)     # [1, 1, 64, time]

        # Інференс
        with torch.no_grad():
            outputs = model(spec)
            _, predicted = torch.max(outputs, 1)
            label = LABELS[predicted.item()]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    # Потім в браузері відкриваєш http://127.0.0.1:5000/
    app.run(host="0.0.0.0", port=5000, debug=False)
