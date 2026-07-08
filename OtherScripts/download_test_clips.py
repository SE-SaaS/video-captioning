# Dev utility (not part of the pipeline): downloads stock clips for local testing.
# Requires yt-dlp + curl_cffi (pip install -U yt-dlp curl_cffi) and ffmpeg on PATH.
# Runs yt-dlp as a module so it uses this Python's curl_cffi (impersonation). Saves to ./videos/.
#
# TLS note: this machine has Avast "Web/Mail Shield" doing TLS interception. It re-signs every
# server certificate with a private root that lives in the Windows cert store but NOT in certifi's
# bundle, so curl_cffi (which uses its own libcurl/OpenSSL) fails with
# "curl: (60) unable to get local issuer certificate".
# Fix WITHOUT weakening TLS: build a CA bundle = certifi + the Windows ROOT/CA stores (which include
# the Avast root), then point curl_cffi/yt-dlp at it via CURL_CA_BUNDLE / SSL_CERT_FILE. Verification
# stays fully on; we just trust the actual root that is signing these connections.
import base64
import os
import ssl
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CA_BUNDLE = os.path.join(HERE, "_ca_bundle.pem")


def build_ca_bundle() -> str:
    """certifi bundle + all Windows root/intermediate CAs -> single PEM. Returns its path."""
    import certifi

    parts = []
    with open(certifi.where(), "rb") as f:
        parts.append(f.read())
    if hasattr(ssl, "enum_certificates"):  # Windows only
        for store in ("ROOT", "CA"):
            try:
                for der, _enc, _trust in ssl.enum_certificates(store):
                    b64 = base64.encodebytes(der).decode("ascii")
                    parts.append(
                        ("-----BEGIN CERTIFICATE-----\n" + b64 + "-----END CERTIFICATE-----\n").encode("ascii")
                    )
            except Exception:
                pass
    with open(CA_BUNDLE, "wb") as f:
        f.write(b"\n".join(parts))
    return CA_BUNDLE


# Make the bundle available to curl_cffi (impersonation) and everything else in this process.
ca = build_ca_bundle()
os.environ["CURL_CA_BUNDLE"] = ca
os.environ["SSL_CERT_FILE"] = ca
os.environ["REQUESTS_CA_BUNDLE"] = ca

videos = {
    "nature_1": "https://www.pexels.com/video/power-of-nature-27692774/",
    "nature_2": "https://www.pexels.com/video/nature-videos-20541921/",
    "urban_1": "https://pixabay.com/videos/traffic-city-cityscape-urban-night-88921/",
    "urban_2": "https://www.pexels.com/video/urban-traffic-jam-in-new-york-city-34118766/",
    "animals_1": "https://www.pexels.com/video/wild-animals-running-5146598/",
    "animals_2": "https://www.pexels.com/video/a-video-footage-of-wild-animals-4962247/",
    "people_1": "https://www.pexels.com/video/people-walking-in-the-city-free-stock-video-18361966/",
    "people_2": "https://www.pexels.com/video/video-of-people-walking-855564/",
    "sports_1": "https://www.pexels.com/video/soccer-game-in-a-stadium-2657257/",
    "sports_2": "https://www.pexels.com/video/football-match-on-stadium-11918917/",
    "food_1": "https://pixabay.com/videos/cooking-cook-food-art-102181/",
    "food_2": "https://www.pexels.com/video/food-chef-kitchen-cooking-4253333/",
    "weather_1": "https://pixabay.com/videos/rain-storm-hurricane-weather-4252/",
    "weather_2": "https://www.pexels.com/video/dramatic-storm-clouds-timelapse-31988889/",
    "tech_1": "https://www.pexels.com/video/digital-data-display-on-screen-28709421/",
    "tech_2": "https://www.pexels.com/video/database-storage-of-a-server-5028622/",
}

for name, url in videos.items():
    print(f"Downloading {name}...")
    subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "--extractor-args", "generic:impersonate",  # bypass Cloudflare anti-bot (needs curl_cffi)
        "-o", f"videos/{name}.%(ext)s",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        url,
    ])

print("Done!")
