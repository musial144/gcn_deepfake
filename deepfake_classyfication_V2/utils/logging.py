import logging

def get_logger(output_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.FileHandler(output_path, mode='a', encoding = 'utf-8')

    return logging.getLogger()