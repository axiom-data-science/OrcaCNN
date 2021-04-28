import os
import shutil

from flask import request, render_template, Response, current_app
from flask import send_from_directory
from flask import Blueprint
from PIL import Image

from io import BytesIO

from PreProcessing import preprocess_chunk_img
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

WIDTH = 640
HEIGHT = 640

upload = Blueprint('upload', __name__)


@upload.route('/upload', methods=['GET', 'POST'])
def upload_file_and_display():
    filename = request.args['filename']
    try:
        preprocess_chunk_img.main(classpath='uploads',
                              resampling=44100, chunks=1, silent=True)
    except Exception as e:
        logger.error(f"Exception: {e}", exc_info=True)
        return render_template('errors/500.html'), 500

    # remove the uploaded file once images are created
    try:
        os.remove("uploads/" + filename)
        shutil.rmtree(current_app.config['AUDIO_FOLDER'])
    except OSError as e:
        logger.error(f"Error: {e.filename} - {e.strerror}", exc_info=True)

    images = []
    for root, dirs, files in os.walk(current_app.config['IMAGE_FOLDER']):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.png'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0 * w / h
            if aspect > 1.0 * WIDTH / HEIGHT:
                width = min(w, WIDTH)
                height = width / aspect
            else:
                height = min(h, HEIGHT)
                width = height * aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })
    return render_template('upload.html', **{
        'images': images
    })


@upload.route('/<path:filename>')
def image(filename):
    try:
        im = Image.open(filename)
        io = BytesIO()
        im.save(io, format='PNG')
        return Response(io.getvalue(), mimetype='image/png')

    except IOError:
        return render_template('errors/404.html'), 404

    return send_from_directory(current_app.config['IMAGE_FOLDER'], filename)
