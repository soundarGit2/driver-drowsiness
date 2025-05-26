from flask import Flask, render_template, request
from detection import detect_drowsiness
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/output', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    video_path = None
    status_msg = None
    confidence_value = None

    if request.method == 'POST':
        video = request.files.get('video')
        if video and video.filename.endswith(('.mp4', '.avi', '.mov')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)

            output_path, final_status, conf = detect_drowsiness(filepath)
            video_path = output_path

            status_msg = final_status
            confidence_value = round(conf,2)  # in percentage

    return render_template('index.html', video_path=video_path,
                           status_msg=status_msg, confidence=confidence_value)

if __name__ == '__main__':
    app.run(debug=True)

