import shutil

from orcacnnapp import create_app
import atexit
from flask import current_app


application = create_app()


def rm_files_atexit():
    with application.app_context():
        shutil.rmtree(current_app.config['PREDICT_FOLDER'], ignore_errors=True)
        shutil.rmtree(current_app.config['IMAGE_FOLDER'], ignore_errors=True)


if __name__ == '__main__':
    application.run(debug=True)
    atexit.register(rm_files_atexit)
