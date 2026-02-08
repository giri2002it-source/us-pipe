from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import traceback

from analysis import process_pdf_for_symbols

# ------------------ APP SETUP ------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ⬅️ Prevent large uploads from killing Render (10MB max)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ------------------ HELPERS ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


# ------------------ ROUTES ------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "🚀 Symbol Detection API Running"})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # ✅ File presence check
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # ✅ Filename check
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # ✅ Only PDF allowed (PNG breaks your pipeline)
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # ✅ Save uploaded file
        file_id = str(uuid.uuid4())
        pdf_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
        file.save(pdf_path)

        # ✅ Run analysis
        total_counts, output_files, model_used, class_names = process_pdf_for_symbols(
            pdf_path=pdf_path,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_FOLDER,
            dpi=150
        )

        # ✅ Cleanup upload
        os.remove(pdf_path)

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
        # 🔥 Full traceback for Render logs
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/output/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


# ------------------ LOCAL DEV ONLY ------------------
if __name__ == "__main__":
    print("🌐 Running locally on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
