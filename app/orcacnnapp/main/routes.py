import os

from flask import request, redirect, url_for, render_template, Blueprint, current_app
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)


@main.route('/')
@main.route('/home')
def home():
    os.makedirs("uploads/", exist_ok=True)
    return render_template('home.html')


@main.route('/home', methods=['GET', 'POST'])
def get_input_image():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        uploaded_file = request.files['audio_file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
                return render_template('errors/415.html'), 415
            uploaded_file.save(os.path.join(
                current_app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload.upload_file_and_display',
                                    filename=filename))
        return render_template('errors/400.html'), 400
    return redirect(request.url)


@main.route('/about')
def about():
    return render_template('about.html')
