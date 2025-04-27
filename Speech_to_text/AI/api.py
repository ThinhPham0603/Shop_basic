from flask import Flask, request, jsonify, render_template
import os, tempfile
import ffmpeg
from infer_ann_hmm import speech2text_rsa

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/infer", methods=["POST"])
def infer():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio']

    try:
        # Save file tạm (webm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            temp_in_path = tmp_in.name
            file.save(temp_in_path)

        # Convert webm → wav
        temp_out_path = temp_in_path.replace(".webm", ".wav")

        (
            ffmpeg
            .input(temp_in_path)
            .output(temp_out_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )

        # Gọi model predict
        result = speech2text_rsa(temp_out_path)

        # Dọn file tạm
        for path in [temp_in_path, temp_out_path]:
            if os.path.exists(path):
                os.remove(path)

        return jsonify(result)

    except Exception as e:
        import traceback
        print("🔥 Error during inference:", traceback.format_exc())
        return jsonify({"error": "Infer failed", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
