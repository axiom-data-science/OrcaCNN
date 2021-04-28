# OrcaCNN Web-Application


### Things to take care of:

- Currentl only supports `.wav` files.
- Put [convert2ycbcr](https://github.com/scikit-image/scikit-image/blob/main/skimage/restoration/_denoise.py#L724)
 : bool, optional 

 - audio files uploaded are stored in the `uploads` folder first and are then picked up by the pre-processing script.

 - is there a way to stop pre-processing scripy mid-way? refresh does not work.

 - THis app uses [blueprints](https://flask.palletsprojects.com/en/1.1.x/blueprints/#blueprints) and [application factories](https://flask.palletsprojects.com/en/1.1.x/patterns/appfactories/)

 - before running docker on port 80, check if your port 80 is free by `sudo lsof -i :80`, if it shows nginx, stop it using `sudo nginx -s stop`

 - use buildkit for docker-compose: `sudo COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build` and have the line `# syntax=docker/dockerfile:1` in your dockerfile



 ### What's changed:

 #### 1. Directory structure:

**Earlier**
```
-- Detection                   
-- Pod-Classification
-- PreProcessing    
-- README.md        
-- assets    
-- requirements.txt
```

**Now**

```
-- app
-- Pod-Classification
-- README.md
-- assets
-- docker-compose.yml
-- nginx
-- run.sh

```

 #### 2. Minor code changes:

 - Introduced `**kwargs` in pre-processing scripts to help take inputs directly from flask scripts.
 - `from tensorflow.keras.models import load_model` in `Detection.predict`

### What's new:

To aid #4, a very simple `flask` web-app has been added with support for:

- Pre-processing acoustic data (`.wav` files, for now)
- Displaying all pre-processed spectrograms and then allowing for prediction using `predict` prompt button
- uWSGI web server
- Nginx as a reverse proxy in front of uWSGI
- Two docker containers for uWSGI app and nginx
- Uses [BuildKIt](https://docs.docker.com/develop/develop-images/build_enhancements/) to build images

The look of webapp:



This is a starting point of how the `orcacnnapp` should function and look like. I beleive there are a lot of functionalities as well as UI changes that can be made. 

Below are a few future tasks I can think of top off my head:

- Display wave of acoustic data (something close to [OrcaAL](https://orcasound.github.io/orcaal/listen))
- After predicting, display start and end times of detected calls, preferably in a `.csv` file
- 