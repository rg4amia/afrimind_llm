"""python -m enterprise.test_convert : conversion docx / pdf / texte en Markdown, et refus propres."""
import io
import zipfile

from enterprise.convert import to_text

def check(name, cond):
    print(("OK  " if cond else "ECHEC"), name)
    assert cond, name

def refused(name, fn, fragment):
    try:
        fn()
    except ValueError as e:
        check(f"{name} : {e}", fragment in str(e))
    else:
        check(name, False)

NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
def p(text, style=None, lst=False):
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:numPr><w:ilvl w:val=\"0\"/></w:numPr>" if lst else "")
    return f'<w:p><w:pPr>{ppr}</w:pPr><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
def table(rows):
    return "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc>{p(c)}</w:tc>" for c in r) + "</w:tr>" for r in rows) + "</w:tbl>"
def docx(body, xml_prefix=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'{xml_prefix}<w:document {NS}><w:body>{body}</w:body></w:document>')
    return buf.getvalue()

# --- texte ---
r = to_text("Politique-congés.md", "Les congés sont de 25 jours.".encode("utf-8"))
check("texte : titre dérivé du nom de fichier", r["title"] == "Politique congés" and r["text"] == "Les congés sont de 25 jours.")
check("texte : encodage Windows (cp1252) reconnu", "é" in to_text("a.txt", "café".encode("cp1252"))["text"])

check("texte : UTF-16 avec marque d'ordre reconnu", to_text("a.txt", "café".encode("utf-16"))["text"] == "café")

# --- docx ---
body = (p("Politique RH", "Heading1") + p("Les congés payés sont de 25 jours.") + p("Détails", "Titre2")
        + p("Demande sur le portail", lst=True) + p("Réponse sous 5 jours", lst=True)
        + table([["Type", "Jours"], ["Congé payé", "25"], ["Paternité", "a|b"]]) + p("Fin."))
r = to_text("rh.docx", docx(body))
t = r["text"]
check("docx : titres en Markdown", t.startswith("# Politique RH\n\nLes congés payés") and "## Détails" in t)
check("docx : listes groupées", "- Demande sur le portail\n- Réponse sous 5 jours" in t)
check("docx : tableau en Markdown avec séparateur", "| Type | Jours |\n| --- | --- |\n| Congé payé | 25 |" in t)
check("docx : le caractère | d'une cellule est neutralisé", "| Paternité | a/b |" in t)
check("docx : source indiquée", r["source"] == "Word")
refused("docx : archive qui n'est pas un docx", lambda: to_text("x.docx", b"pas un zip"), "illisible")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("autre.xml", "<a/>")
refused("docx : zip sans document Word", lambda: to_text("x.docx", buf.getvalue()), "pas un document Word")
refused("docx : entité XML refusée (attaque par expansion)",
        lambda: to_text("x.docx", docx(p("x"), '<!DOCTYPE d [<!ENTITY a "aaaa">]>')), "XML non autorisée")
refused("docx vide", lambda: to_text("x.docx", docx("")), "aucun texte")

# --- pdf ---
def pdf(pages, encrypt=False):
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for i, lines in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        stream = ("BT /F1 12 Tf 14 TL 50 780 Td " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET").encode("cp1252")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)

try:
    import pypdf  # noqa: F401
except ImportError:
    print("SKIP tests PDF : pypdf n'est pas installé (pip install pypdf)")
else:
    page1 = ["Les congés payés sont de 25 jours par an et la demande", "se fait sur le portail RH avant le départ.", "Délai de réponse : 5 jours."]
    r = to_text("rh.pdf", pdf([page1, ["Seconde page : télétravail 2 jours par semaine (accord du responsable)."]]))
    check("pdf : lignes recollées en phrases", "demande se fait sur le portail RH avant le départ." in r["text"])
    check("pdf : accents et plusieurs pages", "congés" in r["text"] and "télétravail 2 jours" in r["text"] and r["source"] == "PDF")
    check("pdf : parenthèses conservées", "(accord du responsable)" in r["text"])
    refused("pdf : sans texte (scan)", lambda: to_text("scan.pdf", pdf([[]])), "OCR")
    refused("pdf : fichier corrompu", lambda: to_text("x.pdf", b"%PDF-1.4 n'importe quoi"), "PDF")
    refused("pdf : pas un pdf du tout", lambda: to_text("x.pdf", b"bonjour"), "PDF")

# --- refus généraux ---
refused("ancien .doc", lambda: to_text("x.doc", b"\xd0\xcf\x11\xe0xxxx"), ".docx")
refused("format inconnu", lambda: to_text("x.exe", b"MZ"), "non pris en charge")
refused("fichier vide", lambda: to_text("x.txt", b""), "vide")
refused("fichier trop gros", lambda: to_text("x.txt", b"a" * 25_000_001), "volumineux")
print("Tous les tests passent.")
