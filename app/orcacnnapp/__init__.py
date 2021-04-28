import os

from flask import Flask

import argparse

import logging
from orcacnnapp.config import Config

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

logging.getLogger('matplotlib.font_manager').disabled = True


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from orcacnnapp.main.routes import main
    from orcacnnapp.upload.routes import upload
    from orcacnnapp.predict.routes import predict_blueprint
    from orcacnnapp.errors.handlers import errors
    app.register_blueprint(main)
    app.register_blueprint(upload)
    app.register_blueprint(predict_blueprint)
    app.register_blueprint(errors)

    return app
