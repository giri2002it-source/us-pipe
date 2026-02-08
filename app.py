from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from analysis import process_pdf_for_symbols

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'png'}


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🚀 Symbol Detection API Running"})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF or PNG allowed"}), 400

        file_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
        file.save(file_path)

        total_counts, output_files, model_used, class_names = process_pdf_for_symbols(
            file_path,
            MODEL_PATH,
            OUTPUT_FOLDER
        )

        os.remove(file_path)

        return jsonify({
            "success": True,
            "model_used": model_used,
            "classes": list(class_names.values()),
            "symbols_per_class": total_counts,
            "total_symbols": sum(total_counts.values()),
            "image_count": len(output_files),
            "images": [
                f"/output/{os.path.relpath(p, OUTPUT_FOLDER)}"
                for p in output_files
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    print("🌐 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
