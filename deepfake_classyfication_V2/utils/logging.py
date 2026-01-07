import logging

"""Procedura tworząca logger do zapisywania informacji o przebiegu trenowania i ewaluacji modelu w pliku oraz na konsoli."""
def get_logger(output_path):
    logger = logging.getLogger("deepfake_logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(output_path, mode='a', encoding = 'utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter) 
    logger.addHandler(console_handler)

    return logger