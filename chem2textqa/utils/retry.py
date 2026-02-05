import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def with_retry(max_attempts: int = 5, min_wait: float = 1, max_wait: float = 60):
    """Decorator for retrying HTTP-related failures with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
        reraise=True,
    )
