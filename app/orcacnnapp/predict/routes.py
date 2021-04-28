import os
import shutil
import atexit

from flask import request, render_template, Response, current_app
from flask import send_from_directory
from flask import Blueprint
from PIL import Image

from io import BytesIO
from Detection import predict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


predict_blueprint = Blueprint('predict_from_model', __name__)
WIDTH = 640
HEIGHT = 640

@predict_blueprint.route('/predict', methods=['GET', 'POST'])
def predict_from_model():
    try:
        predict.predict(model_path='models/checkpointOrcaCNN_detection_adam_cstr_lr5_2x3_str3-41-0.11.h5',
                 test_path='PreProcessed_image/PreProcessed_audio/uploads/')
    except Exception as e:
        logger.error(f"Exception: {e}", exc_info=True)
        return render_template('errors/500.html'), 500

    images = []
    for root, dirs, files in os.walk(current_app.config['PREDICT_FOLDER']):
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
    return render_template('predict.html', **{
        'images': images
    })

@predict_blueprint.route('/<path:filename>')
def image(filename):
    try:
        im = Image.open(filename)
        io = BytesIO()
        im.save(io, format='PNG')
        return Response(io.getvalue(), mimetype='image/png')

    except IOError:
        return render_template('errors/404.html'), 404

    return send_from_directory(current_app.config['PREDICT_FOLDER'], filename)
