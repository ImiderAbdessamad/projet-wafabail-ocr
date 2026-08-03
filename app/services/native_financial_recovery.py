"""Récupération de champs clés depuis le texte natif PDF (filet de sécurité GLM).

Ne remplace pas la vision : n'injecte un candidat que si le champ est absent
des candidats GLM, ou pour forcer 0,00 sur une section de dettes vide.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

from app.schemas.direct_financial_extraction import (
    DirectFinancialCandidate,
    DirectFinancialEvidence,
)
from app.services.financial_normalizer import parse_decimal_amount

_AMOUNT_RE = re.compile(
    r"(?<!\d)(-?\d{1,3}(?:[ \u00a0.]\d{3})*(?:[,.]\d{2})|-?\d+[,.]\d{2})(?!\d)"
)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_txt = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_txt.lower()).strip()


def _amounts_in(text: str) -> list[str]:
    return [m.group(1) for m in _AMOUNT_RE.finditer(text or "")]


def _has_field(candidates: list[DirectFinancialCandidate], code: str) -> bool:
    return any(c.field_code == code for c in candidates)


def _make(
    *,
    field_code: str,
    raw_value: str,
    page_number: int,
    page_type: str,
    label: str,
    role: str,
    nature: str = "DETAIL",
    confidence: float = 0.75,
) -> DirectFinancialCandidate:
    return DirectFinancialCandidate(
        field_code=field_code,
        raw_value=raw_value,
        period="N",
        nature=nature,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=DirectFinancialEvidence(
            page_number=page_number,
            page_type=page_type,  # type: ignore[arg-type]
            raw_label=label[:180],
            column_role=role,  # type: ignore[arg-type]
            source_excerpt=f"{label}|{raw_value}"[:240],
            orientation=0,
        ),
        warnings=["Récupéré depuis texte natif PDF (filet de sécurité)."],
    )


def _recover_ca_from_cpc(text: str, page_number: int) -> DirectFinancialCandidate | None:
    folded = _fold(text)
    if (
        "compte de produits" not in folded
        and "chiffres d affaires" not in folded
        and "ventes de marchandises" not in folded
    ):
        return None

    # Premier montant après « ventes de marchandises » = CA négoce (colonne N)
    idx = folded.find("ventes de marchandises")
    if idx >= 0:
        after_amounts = _amounts_in(folded[idx:])
        if after_amounts:
            best = after_amounts[0]
            if parse_decimal_amount(best) is not None:
                return _make(
                    field_code="CHIFFRE_AFFAIRES",
                    raw_value=best,
                    page_number=page_number,
                    page_type="CPC",
                    label="Ventes de marchandises (en l'état)",
                    role="TOTAL_EXERCICE_N",
                )

    # Fallback : montant après « Chiffres d'affaires »
    idx2 = folded.find("chiffres d affaires")
    if idx2 < 0:
        idx2 = folded.find("chiffre d affaires")
    if idx2 >= 0:
        after_amounts = _amounts_in(folded[idx2:])
        if after_amounts:
            best = after_amounts[0]
            if parse_decimal_amount(best) is not None:
                return _make(
                    field_code="CHIFFRE_AFFAIRES",
                    raw_value=best,
                    page_number=page_number,
                    page_type="CPC",
                    label="Chiffres d'affaires",
                    role="TOTAL_EXERCICE_N",
                )
    return None


def _recover_resultat_exploitation(
    text: str, page_number: int
) -> DirectFinancialCandidate | None:
    folded = _fold(text)
    # Montant juste avant « III. RESULTAT D'EXPLOITATION (I-II) »
    m = re.search(
        r"("
        r"-?\d{1,3}(?:[ \u00a0.]\d{3})*(?:[,.]\d{2})"
        r")\s*(?:\d{1,3}(?:[ \u00a0.]\d{3})*(?:[,.]\d{2})\s*)?"
        r"iii\.?\s*resultat\s*d[' ]?exploitation",
        folded,
    )
    if not m:
        return None
    raw = m.group(1)
    if parse_decimal_amount(raw) is None:
        return None
    return _make(
        field_code="RESULTAT_EXPLOITATION",
        raw_value=raw,
        page_number=page_number,
        page_type="CPC",
        label="III. RESULTAT D'EXPLOITATION (I-II)",
        role="TOTAL_EXERCICE_N",
        nature="SECTION_TOTAL",
        confidence=0.85,
    )


def _section_has_amount_between(text: str, start_pat: str, end_pat: str) -> bool:
    folded = _fold(text)
    start = re.search(start_pat, folded)
    if not start:
        return False
    end = re.search(end_pat, folded[start.end() :])
    chunk = folded[start.end() : start.end() + (end.start() if end else 400)]
    return bool(_AMOUNT_RE.search(chunk))


def _recover_zero_dettes_financieres(
    text: str, page_number: int
) -> DirectFinancialCandidate | None:
    folded = _fold(text)
    if "dettes de financement" not in folded:
        return None
    # Entre l'en-tête dettes de financement et la section suivante : aucun montant
    has_amt = _section_has_amount_between(
        text,
        r"dettes de financement\s*\(?\s*c\s*\)?",
        r"provisions durables|total i\s*\(|ecarts de conversion",
    )
    if has_amt:
        return None
    return _make(
        field_code="DETTES_FINANCIERES",
        raw_value="0,00",
        page_number=page_number,
        page_type="BILAN_PASSIF",
        label="DETTES DE FINANCEMENT (C)",
        role="EXERCICE_N",
        nature="SECTION_TOTAL",
        confidence=0.8,
    )


def _recover_zero_tresorerie_passif(
    text: str, page_number: int
) -> DirectFinancialCandidate | None:
    folded = _fold(text)
    if "tresorerie" not in folded or "passif" not in folded:
        return None
    # Aucun montant entre l'en-tête trésorerie-passif et TOTAL III
    # (les montants après TOTAL III sont le total général).
    m = re.search(
        r"tresorerie\s*-?\s*passif([\s\S]{0,160}?)total iii",
        folded,
    )
    if not m:
        return None
    if _AMOUNT_RE.search(m.group(1)):
        return None
    return _make(
        field_code="TRESORERIE_PASSIF",
        raw_value="0,00",
        page_number=page_number,
        page_type="BILAN_PASSIF",
        label="TOTAL III TRESORERIE-PASSIF",
        role="EXERCICE_N",
        nature="SECTION_TOTAL",
        confidence=0.8,
    )


def _recover_achats_revendus(text: str, page_number: int) -> DirectFinancialCandidate | None:
    folded = _fold(text)
    m = re.search(
        r"achats revendus[^\n]{0,40}?("
        r"-?\d{1,3}(?:[ \u00a0.]\d{3})*(?:[,.]\d{2})"
        r")",
        folded,
    )
    if not m:
        # Label sur ligne précédente
        m = re.search(
            r"achats revendus[\s\S]{0,60}?("
            r"-?\d{1,3}(?:[ \u00a0.]\d{3})*(?:[,.]\d{2})"
            r")",
            folded,
        )
    if not m:
        return None
    raw = m.group(1)
    if parse_decimal_amount(raw) is None:
        return None
    return _make(
        field_code="ACHATS_REVENDUS",
        raw_value=raw,
        page_number=page_number,
        page_type="CPC",
        label="Achats revendus de marchandises",
        role="TOTAL_EXERCICE_N",
    )


def recover_candidates_from_native_text(
    candidates: list[DirectFinancialCandidate],
    native_pages: dict[int, str],
    *,
    page_types: dict[int, str] | None = None,
) -> list[DirectFinancialCandidate]:
    """Ajoute des candidats natifs pour combler les trous GLM évidents."""
    page_types = page_types or {}
    extra: list[DirectFinancialCandidate] = []
    existing_codes = {c.field_code for c in candidates}

    for page_number, text in sorted(native_pages.items()):
        if not text or len(text.strip()) < 40:
            continue
        ptype = page_types.get(page_number, "")
        folded = _fold(text)

        if "CHIFFRE_AFFAIRES" not in existing_codes:
            if ptype in {"CPC", ""} or "compte de produits" in folded:
                ca = _recover_ca_from_cpc(text, page_number)
                if ca is not None:
                    extra.append(ca)
                    existing_codes.add("CHIFFRE_AFFAIRES")

        # Toujours proposer RE natif CPC (I-II) pour battre le conflit ESG
        if ptype in {"CPC", ""} or "resultat d exploitation (i-ii)" in folded:
            re_cand = _recover_resultat_exploitation(text, page_number)
            if re_cand is not None:
                extra.append(re_cand)

        if "DETTES_FINANCIERES" not in existing_codes:
            if ptype in {"BILAN_PASSIF", ""} or "dettes de financement" in folded:
                df = _recover_zero_dettes_financieres(text, page_number)
                if df is not None:
                    extra.append(df)
                    existing_codes.add("DETTES_FINANCIERES")

        if "TRESORERIE_PASSIF" not in existing_codes:
            if ptype in {"BILAN_PASSIF", ""} or "tresorerie" in folded:
                tp = _recover_zero_tresorerie_passif(text, page_number)
                if tp is not None:
                    extra.append(tp)
                    existing_codes.add("TRESORERIE_PASSIF")

        if "ACHATS_REVENDUS" not in existing_codes:
            if "achats revendus" in folded:
                ar = _recover_achats_revendus(text, page_number)
                if ar is not None:
                    extra.append(ar)
                    existing_codes.add("ACHATS_REVENDUS")

    return list(candidates) + extra
