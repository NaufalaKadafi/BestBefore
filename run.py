"""Start BestBefore: python run.py"""

from bestbefore.config import get_settings
from bestbefore.server import create_app

if __name__ == "__main__":
    settings = get_settings()
    print(f"BestBefore on http://localhost:{settings.port}  (Bifrost: {settings.bifrost_url})")
    create_app().run(host="127.0.0.1", port=settings.port, debug=False, threaded=True)
