from main import app
import os

if __name__ == "__main__":
    # Use waitress for production
    from waitress import serve
    port = int(os.environ.get("PORT", "5000"))
    serve(app, host="0.0.0.0", port=port, url_scheme="http", threads=6)
