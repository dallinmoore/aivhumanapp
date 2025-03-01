from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import sys
import time
import traceback  # Add this for detailed error messages
from flask_cors import CORS

# Add model-inference-utility to path to import prediction module
sys.path.append(os.path.join(os.path.dirname(__file__), 'model-inference-utility'))
from predict import predict_single_image

app = Flask(__name__)
CORS(app)
app.secret_key = "your_secret_key"  # Needed for flash messages
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Update the model path to point to the PyTorch model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model-inference-utility', 'models', 'efficientnet.pth')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("app.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        
        # Save the file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run inference on the uploaded image using the predict_single_image function
            label, confidence = predict_single_image(filepath, MODEL_PATH)
            
            # Convert confidence to percentage
            confidence_pct = confidence * 100
            
            # Determine confidence strength description
            if confidence > 0.9:
                strength = "very strong"
            elif confidence > 0.75:
                strength = "strong"
            elif confidence > 0.6:
                strength = "moderate"
            else:
                strength = "weak"
                
            # Create message
            message = f"Analysis complete: This image appears to be {label} " \
                     f"(Confidence: {confidence_pct:.6f}%)"
            
            detailed_message = f"The model has {strength} confidence that this image is {label.lower()}."
            
            if confidence < 0.6:
                note = "Note: This prediction has low confidence and may not be reliable."
            else:
                note = None
            
            # Generate base64 of the image for display
            import base64
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                img_data_uri = f"data:image/jpeg;base64,{img_base64}"
            
            # Delete the original file after processing
            try:
                os.remove(filepath)
                print(f"File {filepath} has been deleted after analysis")
            except Exception as del_error:
                print(f"Error deleting file: {del_error}")
            
            return render_template('app.html', 
                                  message=message,
                                  detailed_message=detailed_message,
                                  note=note,
                                  prediction=label,
                                  probability=f"{confidence_pct:.1f}%",
                                  image_data=img_data_uri)
                                  
        except Exception as e:
            error_details = traceback.format_exc()
            
            # Also delete the file if there's an error
            try:
                os.remove(filepath)
                print(f"File {filepath} has been deleted after error")
            except Exception as del_error:
                print(f"Error deleting file: {del_error}")
                
            return render_template('app.html', 
                                  message=f"Error analyzing image: {str(e)}",
                                  error_details=error_details)
    
    return render_template('app.html', 
                          message="Invalid file format. Please upload a PNG or JPEG image.")

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            label, confidence = predict_single_image(filepath, MODEL_PATH)
            confidence_pct = confidence * 100
            
            response = {
                "prediction": label,
                "confidence": confidence_pct
            }
            
            os.remove(filepath)  # Clean up file after analysis
            return jsonify(response), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)



