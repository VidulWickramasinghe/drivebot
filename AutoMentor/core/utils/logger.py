import logging
import sys

def setup_logging():
    """Configures basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/automentor.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING) # Suppress noisy httpx logs
    logging.getLogger('httpcore').setLevel(logging.WARNING) # Suppress noisy httpcore logs

