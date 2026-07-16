"""Chargement de documents : PDF direct ou archive ZIP contenant des PDF.

Les liasses fiscales arrivent souvent sous forme d'archives ZIP regroupant
plusieurs bilans scannés (un PDF par exercice ou par dépôt). Ce module
décompresse l'archive, liste les entrées PDF et extrait le PDF cible.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import fitz


@dataclass
class ZipPdfEntry:
    """Métadonnées d'un PDF contenu dans une archive ZIP."""

    path: str
    pages: int
    size_bytes: int


class DocumentLoadError(ValueError):
    """Erreur de chargement (format invalide, entrée introuvable…)."""


def _count_pdf_pages(data: bytes) -> int:
    with fitz.open(stream=io.BytesIO(data), filetype="pdf") as doc:
        return len(doc)


def list_zip_pdf_entries(content: bytes) -> list[ZipPdfEntry]:
    """Liste les PDF contenus dans une archive ZIP, triés par nom."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries: list[ZipPdfEntry] = []
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                    continue
                pdf_bytes = archive.read(info.filename)
                if not pdf_bytes.startswith(b"%PDF"):
                    continue
                entries.append(
                    ZipPdfEntry(
                        path=info.filename.replace("\\", "/"),
                        pages=_count_pdf_pages(pdf_bytes),
                        size_bytes=info.file_size,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise DocumentLoadError("Archive ZIP invalide ou corrompue.") from exc

    if not entries:
        raise DocumentLoadError("Aucun fichier PDF trouvé dans l'archive ZIP.")

    return sorted(entries, key=lambda e: e.path)


def extract_pdf_from_zip(content: bytes, entry_path: str | None = None) -> tuple[bytes, str]:
    """Extrait un PDF d'une archive ZIP.

    Si `entry_path` est omis, sélectionne automatiquement le PDF avec le
    plus de pages (souvent la liasse la plus complète dans l'archive).
    """
    entries = list_zip_pdf_entries(content)
    if entry_path:
        normalized = entry_path.replace("\\", "/")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if normalized not in [e.path for e in entries]:
                    # tolérance sur les séparateurs
                    match = next((e.path for e in entries if e.path.endswith(normalized.split("/")[-1])), None)
                    if not match:
                        raise DocumentLoadError(
                            f"Entrée « {entry_path} » introuvable dans le ZIP. "
                            f"Entrées disponibles : {[e.path for e in entries]}"
                        )
                    normalized = match
                data = archive.read(normalized)
                if not data.startswith(b"%PDF"):
                    raise DocumentLoadError(f"L'entrée « {normalized} » n'est pas un PDF valide.")
                return data, normalized.split("/")[-1]
        except zipfile.BadZipFile as exc:
            raise DocumentLoadError("Archive ZIP invalide ou corrompue.") from exc

    # Sélection automatique : PDF avec le plus de pages (liasse la plus complète)
    best = max(entries, key=lambda e: (e.pages, e.size_bytes))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        data = archive.read(best.path)
    return data, best.path.split("/")[-1]


def is_zip_content(content: bytes, filename: str | None, content_type: str | None) -> bool:
    if content[:4] == b"PK\x03\x04":
        return True
    if filename and filename.lower().endswith(".zip"):
        return True
    if content_type in ("application/zip", "application/x-zip-compressed"):
        return True
    return False
