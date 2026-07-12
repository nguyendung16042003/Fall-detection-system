"""Structured logging configuration using structlog.

Provides JSON-formatted logs with event_id tracking for distributed tracing.
"""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor


def add_app_context(_: str, _2: str, event_dict: EventDict) -> EventDict:
    """Add application-specific context to all logs."""
    # Add any global context here
    return event_dict


def configure_logging() -> None:
    """Configure structlog for JSON output with timestamps."""
    structlog.configure(
        processors=[
            # Add log level
            structlog.stdlib.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Add custom app context
            add_app_context,
            # Format exception info if present
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Render as JSON
            structlog.processors.JSONRenderer(),
        ],
        # Use standard library logging under the hood
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        # Context class for storing context
        context_class=dict,
        # Logger factory
        logger_factory=structlog.PrintLoggerFactory(),
        # Cache logger
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with the given name."""
    return structlog.get_logger(name)


# Configure logging on import
configure_logging()
