from flask import Flask, request, jsonify, render_template, send_from_directory
import sys
import os
from werkzeug.utils import secure_filename

# Add the current directory to the Python path to import rag.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Import the function from your existing rag.py file
try:
    from rag import process_document

except ImportError as e:
    print(f"Error importing rag.py: {e}")
    print("Please ensure rag.py is in the same directory and contains process_document.")
    sys.exit(1)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """
    Serves the file upload HTML form from the templates directory.
    """
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """
    Serves static files like CSS.
    """
    return send_from_directory('static', filename)


@app.route('/process_file', methods=['POST'])
def process_file():
    """
    Endpoint to handle file uploads and process them using rag.py.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        original_filename = file.filename
        secured_name = secure_filename(original_filename)

        # Ensure the secured filename retains the original extension
        _, original_ext = os.path.splitext(original_filename)
        if original_ext and not secured_name.lower().endswith(original_ext.lower()):
             secured_name = secured_name + original_ext

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secured_name)
        file.save(filepath)

        try:
 
            report = process_document(filepath)

            # Return the report
            return jsonify({"result": report})

        except Exception as e:

            print(f"An error occurred during file processing: {e}")

            if os.path.exists(filepath):
                 # os.remove(filepath)
                 pass # Keep file for debugging
            return jsonify({"error": f"An internal error occurred while processing the file: {e}"}), 500



if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
