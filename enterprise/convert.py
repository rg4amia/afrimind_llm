"""Conversion de fichiers en texte (Markdown simple) avant indexation dans le RAG.
Formats : .txt .md (lus tels quels), .docx (titres, listes, tableaux), .pdf (texte natif).
Hors périmètre, avec un message clair : PDF scannés (pas d'OCR), anciens .doc, fichiers protégés par mot de passe."""
import io
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

logging.getLogger("pypdf").setLevel(logging.ERROR)   # pypdf bavarde sur les fichiers abîmés ; l'erreur est déjà renvoyée à l'utilisateur

MAX_FILE_BYTES = 25_000_000
MAX_XML_BYTES = 60_000_000       # taille décompressée du contenu d'un .docx (protège des archives piégées)
MAX_PDF_PAGES = 500
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HEADING = re.compile(r"^(?:heading|titre)\s*(\d)$", re.I)

def _decode(data):
    # UTF-16 seulement avec sa marque d'ordre des octets : sans elle, du cp1252 serait pris à tort pour de l'UTF-16
    encodings = ["utf-8-sig"] + (["utf-16"] if data[:2] in (b"\xff\xfe", b"\xfe\xff") else []) + ["cp1252"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("encodage du fichier texte non reconnu")

def _tidy(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ---------- .docx ----------

def _paragraph(p):
    parts = []
    for node in p.iter():
        if node.tag == W + "t":
            parts.append(node.text or "")
        elif node.tag == W + "tab":
            parts.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            parts.append("\n")
    text = "".join(parts).strip()
    if not text:
        return ""
    ppr = p.find(W + "pPr")
    style = ppr.find(W + "pStyle") if ppr is not None else None
    name = (style.get(W + "val") if style is not None else "") or ""
    m = HEADING.match(name.replace("-", " "))
    if m:
        return "#" * min(int(m.group(1)), 6) + " " + text
    if name.lower() in ("title", "titre"):
        return "# " + text
    if ppr is not None and ppr.find(W + "numPr") is not None:
        return "- " + text
    return text

def _table(tbl):
    rows = []
    for tr in tbl.findall(W + "tr"):
        cells = []
        for tc in tr.findall(W + "tc"):
            cell = " ".join(filter(None, (_paragraph(p) for p in tc.iter(W + "p"))))
            cells.append(cell.replace("|", "/").replace("\n", " "))
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if len(rows) > 1:
        rows.insert(1, "| " + " | ".join("---" for _ in rows[0].strip("|").split("|")) + " |")
    return "\n".join(rows)

def _docx(data):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("fichier .docx illisible (archive invalide ou ancien format .doc : enregistrez-le en .docx)")
    try:
        info = z.getinfo("word/document.xml")
    except KeyError:
        raise ValueError("ce fichier n'est pas un document Word (.docx) valide")
    if info.file_size > MAX_XML_BYTES:
        raise ValueError("document trop volumineux une fois décompressé")
    xml = z.read(info)
    head = xml[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise ValueError("document refusé : déclaration XML non autorisée")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise ValueError("contenu du .docx illisible")
    body = root.find(W + "body")
    blocks = []
    for child in body if body is not None else []:
        if child.tag == W + "p":
            blocks.append(_paragraph(child))
        elif child.tag == W + "tbl":
            blocks.append(_table(child))
    # les éléments de liste consécutifs restent groupés ; le reste est séparé par une ligne vide
    out = []
    for b in filter(None, blocks):
        if out and b.startswith("- ") and out[-1].startswith("- "):
            out[-1] += "\n" + b
        else:
            out.append(b)
    return "\n\n".join(out)

# ---------- .pdf ----------

def _reflow(page_text):
    """Les PDF coupent les lignes au milieu des phrases : on les recolle en paragraphes."""
    paras, cur = [], ""
    for line in (l.strip() for l in page_text.splitlines()):
        if not line:
            if cur:
                paras.append(cur); cur = ""
            continue
        if cur and (len(cur) < 40 or re.search(r"[.!?:;]$", cur) or re.match(r"^([-•*]|\d+[.)])\s", line)):
            paras.append(cur); cur = line
        else:
            cur = f"{cur} {line}" if cur else line
    if cur:
        paras.append(cur)
    return "\n\n".join(paras)

def _pdf(data):
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        raise ValueError("lecture des PDF indisponible : installez pypdf (pip install pypdf)")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF protégé par mot de passe : retirez la protection puis réessayez")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"PDF trop long ({len(reader.pages)} pages, maximum {MAX_PDF_PAGES})")
        pages = [_reflow(p.extract_text() or "") for p in reader.pages]
    except PyPdfError:
        raise ValueError("PDF illisible ou corrompu")
    text = "\n\n".join(p for p in pages if p)
    if len(text.strip()) < 20:
        raise ValueError("ce PDF ne contient pas de texte exploitable (scan ou image) : l'OCR n'est pas pris en charge")
    return text

# ---------- point d'entrée ----------

def to_text(filename, data):
    """Renvoie dict(title, text, source, note). Lève ValueError (message affichable) si le fichier est refusé."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"fichier trop volumineux (maximum {MAX_FILE_BYTES // 1_000_000} Mo)")
    if not data:
        raise ValueError("fichier vide")
    path = Path(filename or "document")
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        text, source = _decode(data), "texte"
    elif ext == ".docx":
        text, source = _docx(data), "Word"
    elif ext == ".pdf":
        text, source = _pdf(data), "PDF"
    elif ext == ".doc":
        raise ValueError("ancien format .doc non pris en charge : enregistrez le fichier en .docx")
    else:
        raise ValueError(f"format « {ext or 'inconnu'} » non pris en charge (formats : .txt, .md, .docx, .pdf)")
    text = _tidy(text)
    if not text:
        raise ValueError("aucun texte trouvé dans ce fichier")
    title = re.sub(r"[-_]+", " ", path.stem).strip()[:200] or "Document"
    return dict(title=title, text=text, source=source, chars=len(text))
