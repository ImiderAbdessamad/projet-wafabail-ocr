"""Extraction intelligente des liasses fiscales marocaines (PCGM).

Pipeline :
1. Rapport d'indicateurs structuré → parsing texte direct.
2. Liasse PCGM avec couche texte (ex. DMT FY25) → extraction texte PCGM
   sur toutes les pages (rapide, fiable).
3. Liasse scannée (ZIP, 0 texte) → OCR vision Ollama page par page
   (séquentiel + retries pour éviter les 504 du serveur distant).
4. Échec → `LIASSE_ECHEC`, aucune valeur inventée.

Règle d'or : ne jamais inventer de valeur.
"""
from __future__ import annotations

import re
import unicodedata
from io import BytesIO

import fitz  # pymupdf

from app.schemas.liasse import (
    FinancialElement,
    LiasseExtractionResult,
    RawComponent,
    ScoringInput,
)
from app.services.document_inspector import inspect_document

# Référentiel canonique des 19 éléments financiers (cf. « Formules de calcul »)
ELEMENTS_19: list[tuple[int, str, str, str]] = [
    (1, "ACTIFS_IMMOBILISES", "Actifs immobilisés", "Bilan Actif"),
    (2, "TOTAL_BILAN", "Total du bilan", "Bilan Actif"),
    (3, "CHIFFRE_AFFAIRES", "Chiffre d'affaires", "CPC"),
    (4, "CA_EXPORT", "CA à l'export", "CPC"),
    (5, "DETTES_BANCAIRES_MLT", "Dettes bancaires MLT", "Bilan Passif"),
    (6, "DETTES_BANCAIRES_CT", "Dettes bancaires CT", "Bilan Passif"),
    (7, "PASSIF_CIRCULANT", "Passif circulant", "Bilan Passif"),
    (8, "DETTES_FOURNISSEURS", "Dettes fournisseurs", "Bilan Passif"),
    (9, "COMPTE_COURANT_ASSOCIES", "Compte courant d'associés", "Bilan"),
    (10, "TRESORERIE_PASSIF", "Trésorerie au passif", "Bilan Passif"),
    (11, "ACTIF_CIRCULANT", "Actif circulant", "Bilan Actif"),
    (12, "CREANCES_CLIENTS", "Créances clients", "Bilan Actif"),
    (13, "TRESORERIE_ACTIF", "Trésorerie à l'actif", "Bilan Actif"),
    (14, "CAISSE", "Caisse", "Bilan Actif"),
    (15, "ACHATS_REVENDUS", "Achats revendus", "CPC"),
    (16, "AUTRES_CHARGES", "Autres charges", "CPC"),
    (17, "CHARGES_INTERETS", "Charges d'intérêts", "CPC"),
    (18, "RESULTAT_NET", "Résultat net", "CPC"),
    (19, "TYPE_RESULTAT", "Type de résultat", "Dérivé"),
]

# Postes additionnels nécessaires aux 13 ratios. Ils sont retournés dans
# `raw_components` car le référentiel historique des éléments calculés reste
# volontairement limité aux 19 postes.
SCORING_METRICS: dict[str, tuple[str, str, str]] = {
    "FONDS_PROPRES": ("Fonds propres", "Bilan Passif", "autonomie_financiere, rentabilite_financiere"),
    "CAF": ("Capacité d'autofinancement", "CPC", "capacite_remboursement, caf_sur_ca"),
    "FDR": ("Fonds de roulement", "Bilan", "fdr_sur_ca"),
    "CA_N1": ("Chiffre d'affaires N-1", "CPC", "croissance_ca"),
    "AMORTISSEMENTS": ("Dotations aux amortissements", "CPC", "caf (contrôle)"),
    "ENCOURS_LEASING": ("Encours leasing", "Bilan Passif", "endettement_global_apres_operation"),
    "CMT": ("Crédit moyen terme", "Bilan Passif", "endettement_global_apres_operation"),
    "NOUVEAU_FINANCEMENT": ("Nouveau financement demandé", "Externe", "endettement_global_apres_operation"),
}

_LABEL_TO_ELEMENT = {label: (num, code, source) for num, code, label, source in ELEMENTS_19}

_SOURCES = ("Bilan Actif", "Bilan Passif", "CPC", "Bilan", "Dérivé")
_NOTES = ("Bénéficiaire", "Déficitaire", "Nul")

# Montant PCGM : "47 206 065,50" éventuellement précédé de - ou − (U+2212)
_AMOUNT_RE = r"[-−]?\d{1,3}(?:[   ]\d{3})*(?:,\d{1,2})?"


def _parse_amount(raw: str) -> float:
    cleaned = (
        raw.replace("−", "-")
        .replace(" ", "")
        .replace(" ", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    return float(cleaned)


def _norm(text: str) -> str:
    """Normalise les espaces/apostrophes pour un matching robuste."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("’", "'").replace(" ", " ")


def extract_pdf_text(content: bytes) -> str:
    with fitz.open(stream=BytesIO(content), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def count_pdf_pages(content: bytes) -> int:
    with fitz.open(stream=BytesIO(content), filetype="pdf") as doc:
        return len(doc)


def is_pcgm_native_liasse(text: str) -> bool:
    """Détecte une liasse fiscale PCGM avec couche texte (pas un scan)."""
    lowered = text.lower()
    markers = [
        "bilan (actif)",
        "identification du contribuable",
        "tableau :",
        "compte de produits",
        "liasse fiscale",
    ]
    hits = sum(1 for m in markers if m in lowered)
    return hits >= 2 and len(text.strip()) >= 500


def _find_amount_near_label(
    text: str,
    labels: list[str],
    *,
    gap: int = 500,
    pick: str = "first",
    min_abs: float = 10.0,
) -> float | None:
    """Cherche un montant PCGM dans une fenêtre après un libellé."""
    for label in labels:
        start = 0
        label_lower = label.lower()
        while True:
            idx = text.lower().find(label_lower, start)
            if idx < 0:
                break
            window = text[idx + len(label) : idx + len(label) + gap]
            amounts = [
                _parse_amount(a)
                for a in re.findall(_AMOUNT_RE, window)
            ]
            amounts = [a for a in amounts if abs(a) >= min_abs]
            if amounts:
                if pick == "last":
                    return amounts[-1]
                if pick == "max":
                    return max(amounts, key=abs)
                return amounts[0]
            start = idx + 1
    return None


def _regex_amount(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return _parse_amount(m.group(1))


def _build_result_from_observations(
    observations,
    *,
    metadata: dict[str, str | None],
    pages_total: int | None,
    filename: str | None,
    document_kind: str,
    document_type: str = "financial_statements",
    period_type: str = "actual",
    inspection=None,
) -> LiasseExtractionResult:
    from app.services.accounting_checks import run_accounting_checks
    from app.services.derived_fields import apply_derived_fields
    from app.services.field_resolver import resolve_all_fields
    from app.services.result_builder import build_extraction_result

    resolved = resolve_all_fields(observations)
    resolved = apply_derived_fields(resolved)
    warnings, _, accounting_checks = run_accounting_checks(resolved)
    provenance = {
        code: {
            "selected_value": str(res.selected_value) if res.selected_value is not None else None,
            "calculated_value": str(res.calculated_value) if res.calculated_value is not None else None,
            "detection_status": res.detection_status,
            "confidence": res.confidence,
            "selection_reason": res.selection_reason,
            "validation_status": res.validation_status,
        }
        for code, res in resolved.items()
    }
    sections_detected = {
        "BILAN_ACTIF": any(o.section == "BILAN_ACTIF" for o in observations),
        "BILAN_PASSIF": any(o.section == "BILAN_PASSIF" for o in observations),
        "CPC": any(o.section in {"CPC", "ESG"} for o in observations),
    }
    result = build_extraction_result(
        resolved=resolved,
        metadata=type("Meta", (), metadata)(),
        sections_detected=sections_detected,
        pages_total=pages_total or 0,
        pages_analyzed=pages_total or 0,
        elapsed_ms=0,
        filename=filename,
        document_kind=document_kind,
        document_type=document_type,
        period_type=period_type,
        inspection=inspection,
        extra_warnings=warnings,
        field_provenance=provenance,
        accounting_checks=accounting_checks,
        eligible_for_automatic_scoring=(period_type == "actual"),
        scoring_mode="actual" if period_type == "actual" else "forecast_review",
    )
    return result


def parse_pcgm_native_liasse(
    text: str, filename: str | None = None, pages_total: int | None = None
) -> LiasseExtractionResult:
    """Extraction texte d'une liasse PCGM native (multi-pages, couche texte)."""
    from app.services.native_observation_extractor import (
        extract_metadata_from_native_text,
        extract_native_observations,
    )

    observations = extract_native_observations(text)
    metadata = extract_metadata_from_native_text(text)
    if observations:
        return _build_result_from_observations(
            observations,
            metadata=metadata,
            pages_total=pages_total,
            filename=filename,
            document_kind="LIASSE_NATIVE",
        )

    values: dict[str, float | None] = {}

    values["CHIFFRE_AFFAIRES"] = _find_amount_near_label(
        text,
        ["Chiffres d'affaires", "Ventes de biens et services produits"],
        pick="max",
    )
    values["CA_EXPORT"] = _find_amount_near_label(
        text, ["Ventes à l'export", "CA à l'export"], pick="first", min_abs=1
    )
    values["RESULTAT_NET"] = _find_amount_near_label(
        text, ["Résultat net de l'exercice", "Résultat net"], pick="first"
    )
    values["ACHATS_REVENDUS"] = _regex_amount(
        text, rf"Achats revendus.*?marchandises\s*\n\s*({_AMOUNT_RE})"
    )
    values["CHARGES_INTERETS"] = _find_amount_near_label(
        text, ["Charges d'intérêts", "Charges financières"], pick="first"
    )
    values["AUTRES_CHARGES"] = _find_amount_near_label(
        text, ["Autres charges d'exploitation", "Autres charges"], pick="first"
    )
    values["CA_N1"] = _find_amount_near_label(
        text,
        [
            "Chiffres d'affaires exercice précédent",
            "Chiffres d'affaires N-1",
            "Ventes de biens et services produits",
        ],
        # Les CPC PCGM affichent normalement N puis N-1 : le dernier montant
        # est donc le précédent exercice lorsque les deux sont présents.
        pick="last",
    )
    values["CAF"] = _find_amount_near_label(
        text,
        [
            "Capacité d'autofinancement",
            "CAPACITE D'AUTOFINANCEMENT",
            "CAF",
        ],
        pick="first",
    )
    values["AMORTISSEMENTS"] = _find_amount_near_label(
        text,
        [
            "Dotations d'exploitation",
            "Dotations aux amortissements",
            "Dotations aux amortissements et aux provisions",
        ],
        pick="first",
    )

    values["CREANCES_CLIENTS"] = _find_amount_near_label(
        text, ["Clients et comptes rattachés"], pick="first", gap=200
    )
    values["DETTES_FOURNISSEURS"] = _find_amount_near_label(
        text, ["Fournisseurs et comptes rattachés"], pick="first", gap=200
    )
    values["TRESORERIE_ACTIF"] = _find_amount_near_label(
        text, ["Trésorerie-Actif", "Trésorerie - Actif"], pick="max"
    )
    values["TRESORERIE_PASSIF"] = _find_amount_near_label(
        text, ["Trésorerie-Passif", "Trésorerie - Passif"], pick="max"
    )
    values["CAISSE"] = _find_amount_near_label(text, ["Caisse"], pick="first", min_abs=1, gap=150)
    values["COMPTE_COURANT_ASSOCIES"] = _find_amount_near_label(
        text, ["Comptes d'associés"], pick="first", gap=200
    )

    values["ACTIFS_IMMOBILISES"] = None
    actif_section = text.split("Bilan (passif)")[0] if "bilan (passif)" in text.lower() else text[: len(text) // 2]
    values["ACTIFS_IMMOBILISES"] = _find_amount_near_label(
        actif_section,
        ["IMMOBILISATIONS CORPORELLES (C)", "IMMOBILISATIONS INCORPORELLES (B)"],
        pick="max",
        gap=400,
        min_abs=100,
    )
    values["TOTAL_BILAN"] = _regex_amount(
        text, rf"TOTAL III\s*\n\s*TOTAL III\s*\n\s*({_AMOUNT_RE})"
    )
    values["ACTIF_CIRCULANT"] = _regex_amount(
        text, rf"TOTAL II \(F\+G\+H\+I\)\s*\n\s*TOTAL II \(F\+G\+H\+I\)\s*\n\s*({_AMOUNT_RE})"
    )
    values["PASSIF_CIRCULANT"] = _find_amount_near_label(
        text, ["DETTES DU PASSIF CIRCULANT (F)", "Passif circulant"], pick="max"
    )
    passif_section = text.split("Bilan (passif)")[-1] if "bilan (passif)" in text.lower() else text[len(text) // 2 :]
    values["DETTES_BANCAIRES_MLT"] = None
    m_dette = re.search(
        rf"DETTES DE FINANCEMENT \(C\)(.*?)(?:PROVISIONS DURABLES|ECARTS DE CONVERSION)",
        passif_section,
        re.IGNORECASE | re.DOTALL,
    )
    if m_dette:
        dette_amts = [
            _parse_amount(a)
            for a in re.findall(_AMOUNT_RE, m_dette.group(1))
            if abs(_parse_amount(a)) >= 100
        ]
        if dette_amts:
            values["DETTES_BANCAIRES_MLT"] = max(dette_amts, key=abs)
    values["DETTES_BANCAIRES_CT"] = _find_amount_near_label(
        text, ["Crédits de trésorerie", "Concours bancaires"], pick="first"
    )
    values["FONDS_PROPRES"] = _find_amount_near_label(
        passif_section,
        [
            "Total des capitaux propres",
            "CAPITAUX PROPRES (A)",
            "Total capitaux propres",
        ],
        pick="max",
        gap=350,
    )
    values["FDR"] = _find_amount_near_label(
        text,
        [
            "Fonds de roulement",
            "FONDS DE ROULEMENT",
            "Fonds de roulement fonctionnel",
        ],
        pick="first",
    )
    values["ENCOURS_LEASING"] = _find_amount_near_label(
        text,
        ["Encours leasing", "Crédit-bail", "Crédit bail"],
        pick="first",
    )
    values["CMT"] = _find_amount_near_label(
        text,
        ["Crédit moyen terme", "Crédits à moyen terme", "CMT"],
        pick="first",
    )

    # Nettoyage : supprimer les valeurs manifestement aberrantes (hallucinations regex)
    for code, val in list(values.items()):
        if val is not None and abs(val) > 1e12:
            values[code] = None

    entreprise = None
    block_m = re.search(
        r"Identification du contribuable(.*?)(?:ICE\s*:|Tableau\s*:)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if block_m:
        for line in block_m.group(1).split("\n"):
            line = line.strip()
            if (
                line
                and ":" not in line
                and not re.match(r"^[\d\s.,]+$", line)
                and re.search(r"[A-Za-z]{3,}", line)
                and "conforme" not in line.lower()
            ):
                entreprise = line
                break

    sections = {
        "BILAN_ACTIF": any(
            values.get(c) is not None
            for c in ("ACTIFS_IMMOBILISES", "TOTAL_BILAN", "ACTIF_CIRCULANT", "CREANCES_CLIENTS")
        ),
        "BILAN_PASSIF": any(
            values.get(c) is not None
            for c in ("DETTES_FOURNISSEURS", "PASSIF_CIRCULANT", "DETTES_BANCAIRES_MLT")
        ),
        "CPC": any(
            values.get(c) is not None
            for c in ("CHIFFRE_AFFAIRES", "RESULTAT_NET", "ACHATS_REVENDUS")
        ),
    }

    financial_elements: list[FinancialElement] = []
    found = 0
    for num, code, label, source in ELEMENTS_19:
        if code == "TYPE_RESULTAT":
            rn = values.get("RESULTAT_NET")
            note = None
            if rn is not None:
                note = "Bénéficiaire" if rn > 0 else ("Déficitaire" if rn < 0 else "Nul")
            financial_elements.append(
                FinancialElement(
                    number=num, code=code, label=label, value=None, source=source,
                    note=note, confidence=0.85 if rn is not None else 0.0,
                    detection_status="derived" if rn is not None else "not_detected",
                )
            )
            if rn is not None:
                found += 1
            continue
        val = values.get(code)
        if val is not None:
            found += 1
        financial_elements.append(
            FinancialElement(
                number=num, code=code, label=label, value=val, source=source,
                confidence=0.85 if val is not None else 0.0,
                detection_status="detected" if val is not None else "not_detected",
            )
        )

    treso_a = values.get("TRESORERIE_ACTIF")
    treso_p = values.get("TRESORERIE_PASSIF")
    treso_n = treso_a - treso_p if treso_a is not None and treso_p is not None else None
    raw_components = [
        RawComponent(label=label, value=values[code], source=source, feeds=feeds)
        for code, (label, source, feeds) in SCORING_METRICS.items()
        if values.get(code) is not None
    ]

    scoring_input = ScoringInput(
        chiffre_affaires=values.get("CHIFFRE_AFFAIRES"),
        ca_export=values.get("CA_EXPORT"),
        ca_n1=values.get("CA_N1"),
        total_bilan=values.get("TOTAL_BILAN"),
        fonds_propres=values.get("FONDS_PROPRES"),
        actifs_immobilises=values.get("ACTIFS_IMMOBILISES"),
        actif_circulant=values.get("ACTIF_CIRCULANT"),
        clients=values.get("CREANCES_CLIENTS"),
        fournisseurs=values.get("DETTES_FOURNISSEURS"),
        dettes_financieres=values.get("DETTES_BANCAIRES_MLT"),
        dettes_bancaires_ct=values.get("DETTES_BANCAIRES_CT"),
        passif_circulant=values.get("PASSIF_CIRCULANT"),
        tresorerie_actif=treso_a,
        tresorerie_passif=treso_p,
        tresorerie_nette=treso_n,
        achats=values.get("ACHATS_REVENDUS"),
        frais_financiers=values.get("CHARGES_INTERETS"),
        amortissements=values.get("AMORTISSEMENTS"),
        caf=values.get("CAF"),
        fdr=values.get("FDR"),
        resultat_net=values.get("RESULTAT_NET"),
        compte_courant_associes=values.get("COMPTE_COURANT_ASSOCIES"),
        encours_leasing=values.get("ENCOURS_LEASING"),
        cmt=values.get("CMT"),
        nouveau_financement=values.get("NOUVEAU_FINANCEMENT"),
    )

    completeness = round(100.0 * found / len(ELEMENTS_19), 1)

    return LiasseExtractionResult(
        entreprise=entreprise,
        document_kind="LIASSE_NATIVE",
        elements=financial_elements,
        raw_components=raw_components,
        scoring_input=scoring_input,
        sections_completeness=sections,
        completeness_pct=completeness,
        pages_total=pages_total,
        pages_analyzed=pages_total,
        source_filename=filename,
        warnings=[],
        document_summary=(
            f"Liasse PCGM native {filename or ''} — extraction texte "
            f"({found}/{len(ELEMENTS_19)} éléments, {pages_total or '?'} pages)."
        ),
    )


class LiasseExtractionService:
    """Service d'extraction : PDF -> LiasseExtractionResult."""

    def extract(self, content: bytes, filename: str | None = None) -> LiasseExtractionResult:
        text = _norm(extract_pdf_text(content))

        if len(text.strip()) < 50:
            # Pas de couche texte — sera traité par OCR vision (pipeline principal)
            return LiasseExtractionResult(
                document_kind="LIASSE_ECHEC",
                warnings=["PDF sans couche texte — en attente de l'OCR vision."],
                document_summary=f"Document {filename or ''} scanné.",
            )

        if "Rapport des Éléments Financiers" in text or "Éléments Financiers Calculés" in text:
            return self._parse_indicateurs_report(text)

        return self._parse_generic_liasse(text, filename)

    # ------------------------------------------------------------------
    # Cas 1 : rapport d'indicateurs structuré
    # ------------------------------------------------------------------
    def _parse_indicateurs_report(self, text: str) -> LiasseExtractionResult:
        from app.services.native_observation_extractor import (
            extract_metadata_from_native_text,
            extract_report_observations,
        )

        observations = extract_report_observations(text)
        if observations:
            return _build_result_from_observations(
                observations,
                metadata=extract_metadata_from_native_text(text),
                pages_total=1,
                filename=None,
                document_kind="RAPPORT_INDICATEURS",
                document_type="extraction_report",
                period_type="actual",
            )

        warnings: list[str] = []

        ref_m = re.search(r"Référence liasse\s*:\s*(\S+)", text)
        ent_m = re.search(r"Entreprise\s*:\s*(.+)", text)

        elements = self._parse_elements_table(text)
        raw_components = self._parse_raw_components(text)

        sections = self._assess_sections(elements, raw_components, text, warnings)
        self._apply_confidence(elements, sections)

        scoring_input = self._build_scoring_input(elements, sections)

        complete_count = sum(
            1 for e in elements if e.value is not None and e.confidence > 0
        )
        completeness = round(100.0 * complete_count / len(ELEMENTS_19), 1) if elements else 0.0

        return LiasseExtractionResult(
            reference=ref_m.group(1) if ref_m else None,
            entreprise=ent_m.group(1).strip() if ent_m else None,
            document_kind="RAPPORT_INDICATEURS",
            elements=elements,
            raw_components=raw_components,
            scoring_input=scoring_input,
            sections_completeness=sections,
            completeness_pct=completeness,
            warnings=warnings,
            document_summary=(
                "Rapport d'indicateurs parsé : "
                f"{complete_count}/{len(ELEMENTS_19)} éléments exploitables."
            ),
        )

    def _parse_elements_table(self, text: str) -> list[FinancialElement]:
        """Parse le tableau « # Élément Valeur (MAD) Note Source »."""
        elements: dict[int, FinancialElement] = {}
        sources_alt = "|".join(re.escape(s) for s in _SOURCES)
        notes_alt = "|".join(_NOTES)
        line_re = re.compile(
            rf"^(\d{{1,2}})\s+(.+?)\s+({_AMOUNT_RE})\s*MAD\s*({notes_alt})?\s*({sources_alt})\s*$",
            re.MULTILINE,
        )
        for m in line_re.finditer(text):
            num = int(m.group(1))
            if not 1 <= num <= 19:
                continue
            label = m.group(2).strip()
            canonical = _match_element_label(label)
            if canonical is None:
                continue
            c_num, code, c_source = canonical
            if c_num != num:
                continue
            elements[num] = FinancialElement(
                number=num,
                code=code,
                label=label,
                value=_parse_amount(m.group(3)),
                source=m.group(5) or c_source,
                note=m.group(4),
                detection_status="detected",
            )
        # Toujours retourner les 19 dans l'ordre, même non trouvés
        result = []
        for num, code, label, source in ELEMENTS_19:
            if num in elements:
                result.append(elements[num])
            else:
                result.append(
                    FinancialElement(number=num, code=code, label=label, source=source)
                )
        return result

    def _parse_raw_components(self, text: str) -> list[RawComponent]:
        """Parse les 44 composantes brutes (formats ligne et pipe mélangés)."""
        components: list[RawComponent] = []
        seen: set[str] = set()
        sources_alt = "|".join(re.escape(s) for s in ("Bilan Actif", "Bilan Passif", "CPC", "Bilan"))

        # Format pipe : | Label | 1 234,56 MAD | Source | Alimente |
        pipe_re = re.compile(
            rf"\|\s*([^|]+?)\s*\|\s*({_AMOUNT_RE})\s*MAD\s*\|\s*({sources_alt})\s*\|\s*([^|]+?)\s*\|"
        )
        # Format ligne : Label 1 234,56 MAD Source Alimente
        line_re = re.compile(
            rf"^([A-ZÉÈÀÂÇÎÔÛ][^|\n]*?)\s+({_AMOUNT_RE})\s*MAD\s+({sources_alt})\s+(.+?)\s*$",
            re.MULTILINE,
        )

        # La section 3 commence après « Éléments Bruts de Calcul »
        anchor = text.find("Éléments Bruts de Calcul")
        section = text[anchor:] if anchor >= 0 else text

        for regex in (pipe_re, line_re):
            for m in regex.finditer(section):
                label = re.sub(r"\s+", " ", m.group(1)).strip(" -—")
                if not label or label in seen:
                    continue
                seen.add(label)
                components.append(
                    RawComponent(
                        label=label,
                        value=_parse_amount(m.group(2)),
                        source=m.group(3),
                        feeds=re.sub(r"\s+", " ", m.group(4)).strip(),
                    )
                )
        return components

    def _assess_sections(
        self,
        elements: list[FinancialElement],
        raw_components: list[RawComponent],
        text: str,
        warnings: list[str],
    ) -> dict[str, bool]:
        """Une section est exploitable si au moins une valeur non nulle en provient.

        Les rapports d'extraction remontent des zéros pour les pages non
        capturées : une section entièrement à zéro est traitée comme manquante.
        """
        by_section: dict[str, list[float]] = {
            "BILAN_ACTIF": [],
            "BILAN_PASSIF": [],
            "CPC": [],
        }
        source_map = {"Bilan Actif": "BILAN_ACTIF", "Bilan Passif": "BILAN_PASSIF", "CPC": "CPC"}
        for comp in raw_components:
            key = source_map.get(comp.source)
            if key:
                by_section[key].append(comp.value)
        for el in elements:
            key = source_map.get(el.source)
            if key and el.value is not None:
                by_section[key].append(el.value)

        sections: dict[str, bool] = {}
        labels = {
            "BILAN_ACTIF": "Bilan Actif",
            "BILAN_PASSIF": "Bilan Passif",
            "CPC": "CPC",
        }
        for key, values in by_section.items():
            ok = any(v != 0 for v in values)
            sections[key] = ok
            if not ok:
                warnings.append(
                    f"Section {labels[key]} non capturée par l'OCR : les postes associés "
                    "sont retournés sans valeur (vérification manuelle requise)."
                )
        # Confirmation via la narration du rapport si présente
        if re.search(r"Pages? Bilan Passif non capturé", text, re.IGNORECASE):
            sections["BILAN_PASSIF"] = False
        if re.search(r"Pages? Bilan Actif non capturé", text, re.IGNORECASE):
            sections["BILAN_ACTIF"] = False
        return sections

    def _apply_confidence(
        self, elements: list[FinancialElement], sections: dict[str, bool]
    ) -> None:
        """Confiance par élément selon la complétude de sa section source.

        Les zéros d'une section manquante sont invalidés (value=None) pour
        ne pas polluer les ratios en aval.
        """
        source_map = {"Bilan Actif": "BILAN_ACTIF", "Bilan Passif": "BILAN_PASSIF", "CPC": "CPC"}
        for el in elements:
            if el.source == "Dérivé":
                el.confidence = 0.9 if sections.get("CPC") else 0.0
                continue
            if el.source == "Bilan":  # CCA dépend des deux faces du bilan
                ok = sections.get("BILAN_ACTIF") and sections.get("BILAN_PASSIF")
                el.confidence = 0.85 if ok else 0.0
            else:
                key = source_map.get(el.source, "")
                el.confidence = 0.9 if sections.get(key) else 0.0
            if el.confidence == 0.0:
                el.value = None
                el.note = (el.note or "") or "Section source non capturée"

    def _build_scoring_input(
        self, elements: list[FinancialElement], sections: dict[str, bool]
    ) -> ScoringInput:
        vals: dict[str, float | None] = {
            el.code: el.value for el in elements if el.confidence > 0
        }
        treso_actif = vals.get("TRESORERIE_ACTIF")
        treso_passif = vals.get("TRESORERIE_PASSIF")
        treso_nette = (
            treso_actif - treso_passif
            if treso_actif is not None and treso_passif is not None
            else None
        )
        return ScoringInput(
            chiffre_affaires=vals.get("CHIFFRE_AFFAIRES"),
            ca_export=vals.get("CA_EXPORT"),
            total_bilan=vals.get("TOTAL_BILAN"),
            actifs_immobilises=vals.get("ACTIFS_IMMOBILISES"),
            actif_circulant=vals.get("ACTIF_CIRCULANT"),
            clients=vals.get("CREANCES_CLIENTS"),
            fournisseurs=vals.get("DETTES_FOURNISSEURS"),
            dettes_financieres=vals.get("DETTES_BANCAIRES_MLT"),
            dettes_bancaires_ct=vals.get("DETTES_BANCAIRES_CT"),
            passif_circulant=vals.get("PASSIF_CIRCULANT"),
            tresorerie_actif=treso_actif,
            tresorerie_passif=treso_passif,
            tresorerie_nette=treso_nette,
            achats=vals.get("ACHATS_REVENDUS"),
            frais_financiers=vals.get("CHARGES_INTERETS"),
            resultat_net=vals.get("RESULTAT_NET"),
            compte_courant_associes=vals.get("COMPTE_COURANT_ASSOCIES"),
        )

    # ------------------------------------------------------------------
    # Cas 2 : liasse native — extraction générique par mots-clés
    # ------------------------------------------------------------------
    def _parse_generic_liasse(
        self, text: str, filename: str | None
    ) -> LiasseExtractionResult:
        """Extraction best-effort par libellés PCGM sur un PDF texte quelconque."""
        keyword_map = {
            "CHIFFRE_AFFAIRES": r"(?:Chiffres? d'affaires|Ventes de biens et services)",
            "TOTAL_BILAN": r"Total (?:général de l'actif|du bilan)",
            "RESULTAT_NET": r"Résultat net(?: de l'exercice)?",
            "DETTES_BANCAIRES_MLT": r"Dettes de financement",
            "DETTES_FOURNISSEURS": r"Fournisseurs et comptes rattachés",
            "CREANCES_CLIENTS": r"Clients et comptes rattachés",
        }
        elements: list[FinancialElement] = []
        found = 0
        meta = {num: (code, label, source) for num, code, label, source in ELEMENTS_19}
        code_to_num = {code: num for num, code, _, _ in ELEMENTS_19}
        values: dict[str, float] = {}
        for code, pattern in keyword_map.items():
            m = re.search(rf"{pattern}\D{{0,40}}?({_AMOUNT_RE})", text, re.IGNORECASE)
            if m:
                values[code] = _parse_amount(m.group(1))
                found += 1
        for num, code, label, source in ELEMENTS_19:
            val = values.get(code)
            elements.append(
                FinancialElement(
                    number=num,
                    code=code,
                    label=label,
                    value=val,
                    source=source,
                    confidence=0.6 if val is not None else 0.0,
                )
            )
        treso = None
        scoring_input = ScoringInput(
            chiffre_affaires=values.get("CHIFFRE_AFFAIRES"),
            total_bilan=values.get("TOTAL_BILAN"),
            resultat_net=values.get("RESULTAT_NET"),
            dettes_financieres=values.get("DETTES_BANCAIRES_MLT"),
            fournisseurs=values.get("DETTES_FOURNISSEURS"),
            clients=values.get("CREANCES_CLIENTS"),
            tresorerie_nette=treso,
        )
        return LiasseExtractionResult(
            document_kind="LIASSE_NATIVE",
            elements=elements,
            scoring_input=scoring_input,
            completeness_pct=round(100.0 * found / len(ELEMENTS_19), 1),
            warnings=(
                []
                if found
                else ["Aucun libellé PCGM reconnu dans le texte du document."]
            ),
            document_summary=(
                f"Extraction générique sur {filename or 'document'} : "
                f"{found} poste(s) identifié(s) par mots-clés (confiance moyenne)."
            ),
        )


def _match_element_label(label: str):
    """Matching tolérant du libellé vers le référentiel des 19 éléments."""
    if label in _LABEL_TO_ELEMENT:
        return _LABEL_TO_ELEMENT[label]
    simplified = label.lower().strip()
    for canonical, meta in _LABEL_TO_ELEMENT.items():
        if canonical.lower() == simplified or canonical.lower() in simplified:
            return meta
    return None


def get_liasse_extraction_service() -> LiasseExtractionService:
    return LiasseExtractionService()


async def extract_liasse_document(
    content: bytes,
    filename: str | None,
    service: LiasseExtractionService,
) -> LiasseExtractionResult:
    """Pipeline d'extraction complet pour une liasse fiscale."""
    from app.config import NATIVE_COMPLETENESS_THRESHOLD
    from app.services.vision_ocr import VisionOcrError, extract_liasse_via_vision

    text = _norm(extract_pdf_text(content))
    pages_total = count_pdf_pages(content)
    inspection = inspect_document(content)
    document_type = inspection.document_type
    period_type = inspection.period_type

    if document_type == "forecast_financial_statements":
        native = parse_pcgm_native_liasse(text, filename, pages_total) if text.strip() else LiasseExtractionResult(document_kind="LIASSE_ECHEC")
        native.document_type = document_type
        native.period_type = period_type
        native.eligible_for_automatic_scoring = False
        native.scoring_mode = "forecast_review"
        native.inspection = inspection
        native.scoring_block_reasons = ["Document prévisionnel : scoring automatique réel interdit."]
        return native

    if len(text.strip()) >= 50 and (
        "Rapport des Éléments Financiers" in text
        or "Éléments Financiers Calculés" in text
    ):
        report = service.extract(content, filename)
        report.document_type = document_type
        report.period_type = period_type
        report.eligible_for_automatic_scoring = period_type == "actual"
        report.scoring_mode = "actual" if period_type == "actual" else "forecast_review"
        report.inspection = inspection
        return report

    # Liasse PCGM native (texte) — ex. LIASSE FISCALE DMT FY25.pdf
    if is_pcgm_native_liasse(text):
        native = parse_pcgm_native_liasse(text, filename, pages_total)
        native.document_type = document_type
        native.period_type = period_type
        native.eligible_for_automatic_scoring = period_type == "actual"
        native.scoring_mode = "actual" if period_type == "actual" else "forecast_review"
        native.inspection = inspection
        if native.completeness_pct >= NATIVE_COMPLETENESS_THRESHOLD:
            return native

    # Scannée ou texte insuffisant → OCR vision (séquentiel + retries)
    try:
        vision = await extract_liasse_via_vision(content, filename)
        vision.document_type = document_type
        vision.period_type = period_type
        vision.eligible_for_automatic_scoring = period_type == "actual"
        vision.scoring_mode = "actual" if period_type == "actual" else "forecast_review"
        vision.inspection = inspection
        return vision
    except VisionOcrError as exc:
        # Dernier recours : retourner le résultat texte partiel si disponible
        if is_pcgm_native_liasse(text):
            partial = parse_pcgm_native_liasse(text, filename, pages_total)
            if partial.completeness_pct > 0:
                partial.warnings.insert(
                    0, f"OCR vision échoué ({exc}) — résultat texte partiel conservé."
                )
                return partial
        return LiasseExtractionResult(
            document_kind="LIASSE_ECHEC",
            warnings=[str(exc)],
            source_filename=filename,
            pages_total=pages_total,
            document_summary=f"Échec extraction sur {filename or 'document'} : {exc}",
        )


# Alias rétrocompatible
extract_with_ocr_fallback = extract_liasse_document
