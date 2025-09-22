from __future__ import annotations
import os, time, random, threading
from collections import deque
from typing import Optional

# Load .env if present
try:
    import dotenv; dotenv.load_dotenv()
except Exception:
    pass

# Quiet noisy logs (gRPC/absl)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

class RateLimitedGemini:
    """
    A tiny, threadsafe Gemini client with:
      - request-per-minute limiting (default 15 RPM),
      - hard per-call timeout,
      - robust 429 handling (uses server-provided retry delay if present),
      - exponential backoff with jitter on transient errors.

    Usage:
        client = RateLimitedGemini()
        text = client.generate_text(prompt)
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 rpm_limit: int = 15,
                 timeout_seconds: int = 30,
                 max_retries: int = 6):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Put it in your environment or .env")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.rpm_limit = max(1, int(rpm_limit))
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)

        # rpm limiter state
        self._lock = threading.Lock()
        self._winsize = 60.0  # seconds
        self._req_times = deque()  # monotonic timestamps of recent requests (<= rpm_limit)

        # lazy import/configure SDK
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            self.model,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 256,
                "response_mime_type": "text/plain",
            },
        )

    # ---- rate limit window control ----
    def _acquire_slot(self):
        now = time.monotonic()
        with self._lock:
            # drop entries older than window
            while self._req_times and (now - self._req_times[0] >= self._winsize):
                self._req_times.popleft()
            if len(self._req_times) < self.rpm_limit:
                self._req_times.append(now)
                return 0.0  # no wait
            # need to wait until oldest entry exits the 60s window
            wait = self._winsize - (now - self._req_times[0])
        # release lock before sleep
        time.sleep(max(0.0, wait))
        # try again recursively (it’ll be short after sleep)
        return self._acquire_slot()

    # ---- public call ----
    def generate_text(self, prompt: str) -> str:
        from google.api_core import exceptions as gexc

        attempt = 0
        while True:
            attempt += 1
            # enforce RPM
            self._acquire_slot()
            try:
                resp = self._model.generate_content(
                    prompt,
                    request_options={"timeout": self.timeout_seconds}
                )
                text = getattr(resp, "text", "") or ""
                if not text.strip():
                    # surface candidate info for debugging
                    cand = getattr(resp, "candidates", None)
                    raise RuntimeError(f"Empty response from Gemini. candidates={cand}")
                return text

            except gexc.ResourceExhausted as e:
                # 429 quota exceeded — use server-suggested delay if present, else backoff
                retry_secs = getattr(getattr(e, "retry_delay", None), "seconds", None)
                if retry_secs is None:
                    # exponential backoff with jitter (base ~5s)
                    retry_secs = min(60, 5 * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
                time.sleep(retry_secs)
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Gemini quota exhausted after retries: {e}") from e
                continue

            except (gexc.DeadlineExceeded, gexc.ServiceUnavailable, gexc.InternalServerError) as e:
                # transient server/network — backoff
                sleep = min(60, 2 * (2 ** (attempt - 1))) + random.uniform(0, 1.0)
                time.sleep(sleep)
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Gemini transient failure after retries: {e}") from e
                continue

            except Exception as e:
                # non-retryable or client error
                raise

# Singleton accessor so all callers share the same limiter window
_client_singleton: RateLimitedGemini | None = None
_client_lock = threading.Lock()

def get_client() -> RateLimitedGemini:
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:
                _client_singleton = RateLimitedGemini()
    return _client_singleton
