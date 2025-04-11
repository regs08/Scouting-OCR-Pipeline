import logging

class BaseProcessor:
    def __init__(self, verbose: bool = False, enable_logging: bool = False):
        self.verbose = verbose
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO if self.verbose else logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def display_and_log(self, message: str, data: dict = None):
        """Display and log messages in a uniform manner.

        Args:
            message: The message to display and log.
            data: Optional dictionary of additional data to display.
        """
        # Display the message
        print(message)

        # Log the message if logging is enabled
        if self.enable_logging:
            logging.info(message)

        # If there's additional data, display it
        if data is not None:
            print("Additional Data:")
            for key, value in data.items():
                print(f"{key}: {value}")
                if self.enable_logging:
                    logging.info(f"{key}: {value}")
