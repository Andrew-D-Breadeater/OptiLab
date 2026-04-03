import logging

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
