import argparse
import logging
from dotenv import load_dotenv

# Ensure all plugins are imported so they register themselves

from trackey.core.engine import Engine

from pythonjsonlogger import jsonlogger
from prometheus_client import start_http_server

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler (standard format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(console_handler)

    # File handler (JSON format)
    file_handler = logging.FileHandler("trackey.log")
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser(description="Trackey Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a pipeline configuration")
    run_parser.add_argument("config", help="Path to the config.yaml file")

    args = parser.parse_args()

    if args.command == "run":
        load_dotenv()
        setup_logging()

        # Start Prometheus metrics server
        try:
            start_http_server(8000)
            logging.getLogger("trackey.cli").info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logging.getLogger("trackey.cli").error(f"Failed to start metrics server: {e}")

        logger = logging.getLogger("trackey.cli")
        logger.info(f"Starting Trackey with config: {args.config}")

        try:
            engine = Engine.from_config(args.config)
            engine.run()
        except Exception:
            logger.exception("Fatal error during engine execution")
            raise

if __name__ == "__main__":
    main()
