class Config:
    UPLOAD_FOLDER = 'uploads/'
    IMAGE_FOLDER = 'PreProcessed_image/PreProcessed_audio/uploads/'
    AUDIO_FOLDER = 'PreProcessed_audio/'
    PREDICT_FOLDER = 'pos_orca/'
    secret_key = 'super secret key'
    SESSION_TYPE = 'filesystem'
    SEND_FILE_MAX_AGE_DEFAULT = 0
    UPLOAD_EXTENSIONS = ['.wav', '.WAV']
