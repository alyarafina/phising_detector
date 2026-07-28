"""
feature_extractor.py
---------------------
Mengubah sebuah URL mentah menjadi 30 fitur yang sama persis (nama & urutan)
dengan yang dipakai untuk melatih model pada `phishing.csv`
(UCI Phishing Websites Dataset).

CATATAN PENTING (dibaca dulu sebelum menilai akurasi):
Sebagian fitur asli dataset ini (WebsiteTraffic/Alexa Rank, PageRank Google,
GoogleIndex, LinksPointingToPage/backlink count) awalnya dihitung memakai
layanan pihak ketiga yang sekarang sudah **tidak tersedia lagi secara bebas**
(Alexa resmi ditutup 2022, Google PageRank publik sudah lama dimatikan, query
otomatis ke Google akan diblokir). Untuk fitur-fitur itu, modul ini memakai
**heuristik pendekatan terbaik** (misalnya berdasarkan reachability, status
code, pola redirect) dan bukan nilai asli dari layanan tersebut. Ini didokumentasikan
secara transparan lewat `approximated_features` yang dikembalikan bersama fitur,
dan ditampilkan ke pengguna di aplikasi.

Semua fungsi dibuat DEFENSIF: setiap pemanggilan jaringan (requests/whois/socket)
dibungkus try/except dengan timeout, dan akan fallback ke nilai netral (0) atau
nilai "aman" jika gagal/timeout, supaya aplikasi TIDAK PERNAH crash walau:
- URL tidak valid
- Situs tidak bisa diakses / timeout
- WHOIS tidak tersedia untuk domain tersebut
- Tidak ada koneksi internet sama sekali
"""

import re
import socket
import ssl
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import whois  # python-whois
    WHOIS_AVAILABLE = True
except Exception:
    WHOIS_AVAILABLE = False

try:
    import tldextract
    # Paksa pakai snapshot Public Suffix List bawaan (offline), supaya tidak ada
    # percobaan koneksi ke publicsuffix.org setiap kali extract_features() dipanggil
    # (lebih cepat & tidak gagal kalau jaringan terbatas/diblokir firewall kantor, dll).
    _TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
    TLDEXTRACT_AVAILABLE = True
except Exception:
    TLDEXTRACT_AVAILABLE = False

REQUEST_TIMEOUT = 6
# WHOIS (dan sebagian resolusi DNS) di python-whois memakai raw socket tanpa
# parameter timeout eksplisit. Kita set default timeout socket secara global
# supaya lookup yang lambat/nyangkut tidak membuat aplikasi menggantung lama.
socket.setdefaulttimeout(REQUEST_TIMEOUT)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 PhishingDetectorBot/1.0"
)

SHORTENER_SERVICES = re.compile(
    r"(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|post\.ly|"
    r"just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|cutt\.ly|rebrand\.ly)",
    re.IGNORECASE,
)

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "webscr", "ebayisapi", "suspend", "urgent",
]


class FeatureExtractionResult:
    """Wadah hasil ekstraksi: fitur final + metadata transparansi."""

    def __init__(self, features: dict, approximated_features: list, fetch_ok: bool, error: str = None):
        self.features = features
        self.approximated_features = approximated_features
        self.fetch_ok = fetch_ok
        self.error = error


def _safe_get(url):
    """GET request yang aman: return (response_or_None, ok_bool)."""
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT},
            allow_redirects=True, verify=True,
        )
        return resp, True
    except Exception:
        try:
            # Coba lagi tanpa verifikasi SSL (banyak situs phishing punya cert bermasalah,
            # kita tetap ingin bisa menganalisis kontennya)
            resp = requests.get(
                url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT},
                allow_redirects=True, verify=False,
            )
            return resp, True
        except Exception:
            return None, False


def _get_domain_parts(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if TLDEXTRACT_AVAILABLE:
        ext = _TLD_EXTRACTOR(url)
        registered_domain = ".".join(p for p in [ext.domain, ext.suffix] if p)
        subdomain = ext.subdomain
    else:
        parts = hostname.split(".")
        registered_domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
    return parsed, hostname, registered_domain, subdomain


def _is_ip(hostname):
    try:
        socket.inet_aton(hostname)
        return True
    except Exception:
        return bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname or ""))


def extract_features(raw_url: str) -> FeatureExtractionResult:
    """Fungsi utama: URL string -> FeatureExtractionResult(features=dict 30 kolom, ...)."""

    approximated = []
    url = raw_url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed, hostname, registered_domain, subdomain = _get_domain_parts(url)
    features = {}

    # 1. UsingIP -----------------------------------------------------------
    features["UsingIP"] = -1 if _is_ip(hostname) else 1

    # 2. LongURL -------------------------------------------------------------
    length = len(url)
    features["LongURL"] = 1 if length < 54 else (0 if length <= 75 else -1)

    # 3. ShortURL --------------------------------------------------------
    features["ShortURL"] = -1 if SHORTENER_SERVICES.search(url) else 1

    # 4. Symbol@ -----------------------------------------------------------
    features["Symbol@"] = -1 if "@" in url else 1

    # 5. Redirecting// -------------------------------------------------------
    last_double_slash = url.rfind("//")
    features["Redirecting//"] = -1 if last_double_slash > 7 else 1

    # 6. PrefixSuffix- -------------------------------------------------------
    features["PrefixSuffix-"] = -1 if "-" in (hostname or "") else 1

    # 7. SubDomains ----------------------------------------------------------
    dot_count = registered_domain.count(".") + (1 if subdomain else 0)
    n_sub_dots = hostname.count(".") - registered_domain.count(".") - 1 if hostname else -1
    n_dots_total = hostname.count(".") if hostname else 0
    if n_dots_total <= 1:
        features["SubDomains"] = 1
    elif n_dots_total == 2:
        features["SubDomains"] = 0
    else:
        features["SubDomains"] = -1

    # 8. HTTPS -----------------------------------------------------------------
    features["HTTPS"] = 1 if parsed.scheme == "https" else -1

    # 9. HTTPSDomainURL: kata "https" muncul di bagian domain (bukan protokol) -> mencurigakan
    features["HTTPSDomainURL"] = -1 if "https" in (hostname or "").lower() else 1

    # 10. NonStdPort ----------------------------------------------------------
    port = parsed.port
    features["NonStdPort"] = -1 if (port not in (None, 80, 443)) else 1

    # ---- Fitur yang butuh WHOIS (domain registration & umur domain) ----
    domain_age_days = None
    domain_reg_days = None
    whois_ok = False
    if WHOIS_AVAILABLE:
        try:
            w = whois.whois(registered_domain)

            def _first(v):
                if isinstance(v, list):
                    return v[0] if v else None
                return v

            created = _first(w.creation_date)
            expires = _first(w.expiration_date)
            if isinstance(created, datetime):
                domain_age_days = (datetime.now() - created).days
            if isinstance(created, datetime) and isinstance(expires, datetime):
                domain_reg_days = (expires - created).days
            whois_ok = created is not None
        except Exception:
            whois_ok = False

    # 11. DomainRegLen: registrasi >= 1 tahun -> aman (1), lebih pendek -> phishing (-1)
    if domain_reg_days is not None:
        features["DomainRegLen"] = 1 if domain_reg_days >= 365 else -1
    else:
        features["DomainRegLen"] = -1
        approximated.append("DomainRegLen (WHOIS tidak tersedia, default ke -1/mencurigakan)")

    # 24. AgeofDomain: umur domain >= 6 bulan -> aman (1)
    if domain_age_days is not None:
        features["AgeofDomain"] = 1 if domain_age_days >= 180 else -1
    else:
        features["AgeofDomain"] = -1
        approximated.append("AgeofDomain (WHOIS tidak tersedia, default ke -1/mencurigakan)")

    # 18. AbnormalURL: identitas WHOIS ada & host cocok -> 1, sebaliknya -> -1
    features["AbnormalURL"] = 1 if whois_ok else -1
    if not whois_ok:
        approximated.append("AbnormalURL (WHOIS tidak tersedia)")

    # 25. DNSRecording: domain punya DNS record -> 1, else -1
    try:
        socket.gethostbyname(hostname)
        features["DNSRecording"] = 1
    except Exception:
        features["DNSRecording"] = -1

    # ---- Fitur yang butuh fetch halaman (HTML/JS) ----
    resp, fetch_ok = _safe_get(url)
    html = resp.text if (fetch_ok and resp is not None) else ""
    soup = None
    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None

    def _domain_of(link):
        try:
            d = urlparse(link).hostname or ""
            return d.lower()
        except Exception:
            return ""

    base_domain = (hostname or "").lower()

    if soup is not None:
        # 12. Favicon: favicon dimuat dari domain eksternal -> -1
        favicon_tag = soup.find("link", rel=lambda v: v and "icon" in v.lower())
        if favicon_tag and favicon_tag.get("href"):
            fav_domain = _domain_of(favicon_tag["href"])
            features["Favicon"] = -1 if (fav_domain and base_domain not in fav_domain and fav_domain not in base_domain) else 1
        else:
            features["Favicon"] = 1

        # 13. RequestURL: % elemen (img/script/link) dari domain eksternal
        tags = soup.find_all(["img", "script", "audio", "embed", "iframe"])
        ext_count, total = 0, 0
        for t in tags:
            src = t.get("src")
            if src:
                total += 1
                d = _domain_of(src)
                if d and base_domain not in d and d not in base_domain:
                    ext_count += 1
        if total > 0:
            pct = ext_count / total * 100
            features["RequestURL"] = 1 if pct < 22 else (0 if pct <= 61 else -1)
        else:
            features["RequestURL"] = 1

        # 14. AnchorURL: % <a href> menuju domain lain / kosong / javascript:void
        anchors = soup.find_all("a")
        bad_anchor, total_a = 0, 0
        for a in anchors:
            href = a.get("href")
            if href is None:
                continue
            total_a += 1
            href_l = href.strip().lower()
            if href_l in ("#", "") or href_l.startswith("javascript:") or href_l.startswith("#"):
                bad_anchor += 1
            else:
                d = _domain_of(href)
                if d and base_domain not in d and d not in base_domain:
                    bad_anchor += 1
        if total_a > 0:
            pct = bad_anchor / total_a * 100
            features["AnchorURL"] = 1 if pct < 31 else (0 if pct <= 67 else -1)
        else:
            features["AnchorURL"] = 1

        # 15. LinksInScriptTags: % link di tag <script>/<link>/<meta> dari domain lain
        meta_link_script = soup.find_all(["meta", "script", "link"])
        ext_ls, total_ls = 0, 0
        for t in meta_link_script:
            src = t.get("src") or t.get("href")
            if src:
                total_ls += 1
                d = _domain_of(src)
                if d and base_domain not in d and d not in base_domain:
                    ext_ls += 1
        if total_ls > 0:
            pct = ext_ls / total_ls * 100
            features["LinksInScriptTags"] = 1 if pct < 17 else (0 if pct <= 81 else -1)
        else:
            features["LinksInScriptTags"] = 1

        # 16. ServerFormHandler: form action kosong / about:blank / domain lain -> mencurigakan
        forms = soup.find_all("form")
        if not forms:
            features["ServerFormHandler"] = 1
        else:
            suspicious = False
            neutral = False
            for f in forms:
                action = (f.get("action") or "").strip().lower()
                if action in ("", "about:blank"):
                    suspicious = True
                elif action.startswith("http"):
                    d = _domain_of(action)
                    if d and base_domain not in d and d not in base_domain:
                        neutral = True
            features["ServerFormHandler"] = -1 if suspicious else (0 if neutral else 1)

        # 17. InfoEmail: memakai mailto:/mail() di source -> -1
        features["InfoEmail"] = -1 if ("mailto:" in html.lower() or "mail(" in html.lower()) else 1

        # 20. StatusBarCust: ada onmouseover yang mengubah status bar -> -1
        features["StatusBarCust"] = -1 if re.search(r"onmouseover\s*=", html, re.IGNORECASE) else 1

        # 21. DisableRightClick: event.button==2 / contextmenu di-disable -> -1
        features["DisableRightClick"] = -1 if re.search(r"event\.button\s*==\s*2|oncontextmenu", html, re.IGNORECASE) else 1

        # 22. UsingPopupWindow: window.open dengan input text -> -1
        features["UsingPopupWindow"] = -1 if re.search(r"alert\(|window\.open\s*\(", html, re.IGNORECASE) else 1

        # 23. IframeRedirection: memakai <iframe> -> -1
        features["IframeRedirection"] = -1 if soup.find("iframe") else 1
    else:
        # Halaman tidak bisa diambil (offline / diblokir / timeout) -> fallback netral/mencurigakan
        content_features = [
            "Favicon", "RequestURL", "AnchorURL", "LinksInScriptTags",
            "ServerFormHandler", "InfoEmail", "StatusBarCust", "DisableRightClick",
            "UsingPopupWindow", "IframeRedirection",
        ]
        for cf in content_features:
            features[cf] = 0
        approximated.append("Fitur berbasis konten HTML (Favicon, RequestURL, AnchorURL, dll.) "
                             "di-set netral (0) karena halaman tidak berhasil diakses")

    # 19. WebsiteForwarding: jumlah redirect
    if fetch_ok and resp is not None:
        n_redirects = len(resp.history)
        features["WebsiteForwarding"] = 1 if n_redirects == 0 else (0 if n_redirects <= 3 else -1)
    else:
        features["WebsiteForwarding"] = 0
        approximated.append("WebsiteForwarding (halaman tidak berhasil diakses, default netral)")

    # ---- Fitur yang bergantung layanan pihak ketiga yang sudah tidak tersedia ----
    # 26. WebsiteTraffic: dulu berbasis Alexa Rank (sudah tutup). Pendekatan: situs yang
    # merespons dengan sukses (status 2xx) dianggap punya traffic (1), selain itu netral (0).
    if fetch_ok and resp is not None and 200 <= resp.status_code < 400:
        features["WebsiteTraffic"] = 1
    else:
        features["WebsiteTraffic"] = 0
    approximated.append("WebsiteTraffic (layanan Alexa Rank asli sudah tidak beroperasi; "
                         "didekati dari reachability situs)")

    # 27. PageRank: layanan Google PageRank publik sudah lama dimatikan -> netral
    features["PageRank"] = 0
    approximated.append("PageRank (Google PageRank publik API sudah tidak tersedia sejak lama)")

    # 28. GoogleIndex: query otomatis ke Google akan diblokir/rate-limited -> netral
    features["GoogleIndex"] = 0
    approximated.append("GoogleIndex (query otomatis ke Google tidak dapat dilakukan secara andal)")

    # 29. LinksPointingToPage: dulu berbasis layanan backlink checker berbayar -> netral
    features["LinksPointingToPage"] = 0
    approximated.append("LinksPointingToPage (layanan backlink checker gratis tidak tersedia)")

    # 30. StatsReport: heuristik kata kunci mencurigakan pada URL sebagai pendekatan
    # dari basis data phishing statistik asli (PhishTank/StopBadware, sudah tidak dipakai lagi)
    url_lower = url.lower()
    kw_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    features["StatsReport"] = -1 if (kw_hits >= 2 or _is_ip(hostname)) else 1
    approximated.append("StatsReport (didekati dari heuristik kata kunci mencurigakan pada URL, "
                         "bukan dari basis data PhishTank/StopBadware asli)")

    # Pastikan urutan & kelengkapan fitur sesuai feature_columns.json
    return FeatureExtractionResult(
        features=features,
        approximated_features=approximated,
        fetch_ok=fetch_ok,
        error=None if fetch_ok else "Halaman tidak berhasil diakses (timeout/offline/diblokir).",
    )
