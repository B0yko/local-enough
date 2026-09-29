#!/usr/bin/env python3
"""Seeded, offline generators for local-enough's bundled task datasets.

No LLM is involved: every record is built from hand-written sentence templates and
value pools with ``random.Random(seed)``, so gold labels are exact by construction and
regeneration with the same seed is byte-identical. BANKING77 (classification) is the
one exception: it is downloaded once from the PolyAI repository and sampled with the
seed; every other task is fully synthetic.

Usage:
    uv run python scripts/build_datasets.py --seed 7 [--out DIR] [--banking77-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from local_enough.tasks.base import DatasetCard, FieldSpec, TaskSpec  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "src" / "local_enough" / "data" / "datasets"
DEFAULT_EXAMPLES_OUT = REPO_ROOT / "examples" / "custom-task"
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
BANKING77_TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
BANKING77_LICENSE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/LICENSE"
BANKING77_TRAIN_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
BANKING77_TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"

# --------------------------------------------------------------------------------------
# Shared pools and helpers
# --------------------------------------------------------------------------------------

# ~100 well-known real brands: generated names are checked against this list so no
# fictional pool entry can accidentally collide with a real company.
BLOCKLIST_BRANDS = [
    "google",
    "alphabet",
    "microsoft",
    "amazon",
    "apple",
    "meta",
    "facebook",
    "netflix",
    "tesla",
    "spacex",
    "ibm",
    "oracle",
    "sap",
    "salesforce",
    "adobe",
    "intel",
    "amd",
    "nvidia",
    "samsung",
    "sony",
    "lg",
    "huawei",
    "xiaomi",
    "dell",
    "hp",
    "hewlett-packard",
    "lenovo",
    "cisco",
    "vmware",
    "servicenow",
    "workday",
    "zoom",
    "slack",
    "atlassian",
    "shopify",
    "stripe",
    "paypal",
    "visa",
    "mastercard",
    "americanexpress",
    "coca-cola",
    "pepsi",
    "pepsico",
    "nestle",
    "unilever",
    "procter&gamble",
    "p&g",
    "walmart",
    "target",
    "costco",
    "ikea",
    "nike",
    "adidas",
    "puma",
    "reebok",
    "underarmour",
    "mcdonalds",
    "burgerking",
    "kfc",
    "starbucks",
    "subway",
    "dominos",
    "pizzahut",
    "fedex",
    "ups",
    "dhl",
    "maersk",
    "boeing",
    "airbus",
    "lockheedmartin",
    "generalelectric",
    "ge",
    "siemens",
    "bosch",
    "3m",
    "caterpillar",
    "johndeere",
    "toyota",
    "honda",
    "ford",
    "generalmotors",
    "gm",
    "volkswagen",
    "bmw",
    "mercedes-benz",
    "audi",
    "renault",
    "hyundai",
    "kia",
    "chevron",
    "exxonmobil",
    "exxon",
    "shell",
    "bp",
    "totalenergies",
    "deloitte",
    "pwc",
    "ey",
    "kpmg",
    "mckinsey",
    "accenture",
    "goldmansachs",
    "jpmorgan",
    "citigroup",
    "hsbc",
    "barclays",
    "santander",
    "ups",
    "airbnb",
    "uber",
    "lyft",
    "linkedin",
    "twitter",
    "x-corp",
    "tiktok",
    "bytedance",
    "snapchat",
    "pinterest",
    "reddit",
    "wikipedia",
    "ebay",
    "alibaba",
    "tencent",
    "baidu",
    "spotify",
    "disney",
    "warnerbros",
    "universal",
    "sony-pictures",
    "verizon",
    "at&t",
    "tmobile",
    "vodafone",
    "orange",
    "bt",
    "sky",
    "samsungelectronics",
    "canon",
    "nikon",
    "panasonic",
]


def _normalize_brand(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


BLOCKLIST_NORM = {_normalize_brand(b) for b in BLOCKLIST_BRANDS}


def check_no_blocklisted_brand(name: str) -> None:
    """Word-boundary check: exact-token (or adjacent-token) collisions only, no bare substrings."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    for size in (1, 2, 3):
        for i in range(len(words) - size + 1):
            token = "".join(words[i : i + size])
            if token in BLOCKLIST_NORM:
                raise AssertionError(f"generated name {name!r} collides with blocklisted brand {token!r}")


FIRST_NAMES = [
    "Priya",
    "Marcus",
    "Elena",
    "Tomasz",
    "Aisha",
    "Noah",
    "Freya",
    "Kwame",
    "Sofia",
    "Liam",
    "Ingrid",
    "Dmitri",
    "Hana",
    "Callum",
    "Yuki",
    "Rosa",
    "Declan",
    "Mei",
    "Owen",
    "Zara",
    "Felix",
    "Amara",
    "Bjorn",
    "Nadia",
    "Theo",
    "Layla",
    "Gareth",
    "Ines",
    "Milo",
    "Chiara",
    "Anders",
    "Fatima",
    "Jonas",
    "Wren",
    "Tariq",
    "Esme",
    "Rafael",
    "Ottilie",
    "Kenji",
    "Maya",
]
LAST_NAMES = [
    "Hartley",
    "Novak",
    "Oduya",
    "Berglund",
    "Castellano",
    "Whitfield",
    "Moreau",
    "Lindqvist",
    "Okafor",
    "Bergman",
    "Sorensen",
    "Delacroix",
    "Marchetti",
    "Kowalski",
    "Renner",
    "Abara",
    "Fenwick",
    "Larkin",
    "Vasquez",
    "Holloway",
    "Sato",
    "Brandt",
    "Costa",
    "Nakamura",
    "Hendricks",
    "Solberg",
    "Iqbal",
    "Tremblay",
    "Weiss",
    "Duarte",
    "Ashworth",
    "Callahan",
    "Ferraro",
    "Mbeki",
    "Santini",
    "Boucher",
    "Ravensworth",
    "Adeyemi",
    "Lindgren",
    "Petrova",
]

COMPANY_PREFIXES = [
    "Harrow",
    "Kestrel",
    "Brightfield",
    "Thorncliff",
    "Meridian",
    "Oakstead",
    "Silverline",
    "Cobalt",
    "Larkspur",
    "Fenwick",
    "Ashgrove",
    "Northgate",
    "Amberwood",
    "Foxglove",
    "Hazelmere",
    "Sterling Vale",
    "Copperfield",
    "Elmridge",
    "Ravenscroft",
    "Wrenfield",
    "Moorbank",
    "Sandpiper",
    "Briarwood",
    "Cinderford",
    "Thistledown",
]
SECTOR_SUFFIXES: dict[str, list[str]] = {
    "software": ["Software", "Systems", "Digital", "Data Labs", "Cloud Works"],
    "office supplies": ["Office Supplies", "Stationery Co", "Workplace Goods", "Paper & Print"],
    "logistics": ["Logistics", "Freight", "Distribution", "Haulage"],
    "facilities": ["Facilities", "Maintenance Group", "Property Services", "Site Services"],
}
SECTORS = list(SECTOR_SUFFIXES)
LEGAL_FORMS = {"GB": ["Ltd", "LLP"], "US": ["Inc.", "LLC"], "CA": ["Inc.", "Ltd."]}
COUNTRY_NAMES = {"GB": "United Kingdom", "US": "United States", "CA": "Canada"}

STREETS = [
    "Elm Street",
    "Victoria Road",
    "Kings Parade",
    "Mill Lane",
    "Harbour View",
    "Chapel Row",
    "Foundry Street",
    "Orchard Close",
    "Station Approach",
    "Bridge Street",
    "Maple Avenue",
    "Riverside Drive",
    "Commerce Way",
    "Union Street",
    "Wellington Road",
]
CITIES = {
    "GB": ["London", "Manchester", "Bristol", "Leeds", "Glasgow", "Birmingham", "Cardiff"],
    "US": ["Austin", "Denver", "Seattle", "Boston", "Chicago", "Atlanta", "Portland"],
    "CA": ["Toronto", "Vancouver", "Calgary", "Ottawa", "Winnipeg"],
}
US_AREA_CODES = ["202", "212", "312", "404", "415", "512", "617", "213", "305"]
CA_AREA_CODES = ["416", "604", "403", "613", "204"]

PRODUCT_LINES = [
    "Nimbus Workspace",
    "Ledgerline ERP",
    "ClearRoute TMS",
    "DeskHarbor Furniture",
    "GuardPoint FM",
    "SwiftDock WMS",
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def write_yaml_task(path: Path, spec: TaskSpec) -> None:
    import yaml

    data = spec.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=False, default_flow_style=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_date_header(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}"


def iso(d: date) -> str:
    return d.isoformat()


# --- date-expression resolution: the generator both renders the phrase and computes the
# gold ISO date with the same logic, so extraction gold is exact by construction. ---


def resolve_next_weekday(ref: date, weekday_name: str) -> date:
    target = WEEKDAYS.index(weekday_name)
    days_ahead = (target - ref.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return ref + timedelta(days=days_ahead)


def resolve_this_weekday(ref: date, weekday_name: str) -> date:
    target = WEEKDAYS.index(weekday_name)
    days_ahead = (target - ref.weekday()) % 7
    return ref + timedelta(days=days_ahead)


def resolve_on_day_of_month(ref: date, day: int) -> date:
    y, m = ref.year, ref.month
    cand: date | None
    try:
        cand = date(y, m, day)
    except ValueError:
        cand = None
    if cand is None or cand <= ref:
        m2, y2 = (m + 1, y) if m < 12 else (1, y + 1)
        cand = date(y2, m2, day)
    return cand


def meeting_date_expression(rng: random.Random, ref: date) -> tuple[str, str | None]:
    """Return (phrase-or-empty, iso-date-or-None) for a requested meeting date."""
    choice = rng.choice(["next_weekday", "this_weekday", "in_weeks", "in_days", "on_day", "tomorrow", "none"])
    if choice == "next_weekday":
        wd = rng.choice(WEEKDAYS[:5])
        return f"next {wd}", iso(resolve_next_weekday(ref, wd))
    if choice == "this_weekday":
        wd = rng.choice(WEEKDAYS[:5])
        return f"this {wd}", iso(resolve_this_weekday(ref, wd))
    if choice == "in_weeks":
        n = rng.choice([1, 2, 3])
        word = {1: "a week", 2: "two weeks", 3: "three weeks"}[n]
        return f"in {word}", iso(ref + timedelta(weeks=n))
    if choice == "in_days":
        n = rng.choice([3, 5, 10])
        return f"in {n} days", iso(ref + timedelta(days=n))
    if choice == "on_day":
        day = rng.randint(1, 28)
        return f"on the {ordinal(day)}", iso(resolve_on_day_of_month(ref, day))
    if choice == "tomorrow":
        return "tomorrow", iso(ref + timedelta(days=1))
    return "", None


def uk_phone(rng: random.Random) -> tuple[str, str]:
    """Return (display text, E.164) for a fictional UK mobile in the 07700 900xxx range."""
    last3 = rng.randint(0, 999)
    national = f"07700 900{last3:03d}"
    e164 = f"+447700900{last3:03d}"
    style = rng.choice(["national_spaced", "national_plain", "intl_spaced", "intl_zero", "brackets", "dashed"])
    if style == "national_spaced":
        text = national
    elif style == "national_plain":
        text = national.replace(" ", "")
    elif style == "intl_spaced":
        text = f"+44 7700 900{last3:03d}"
    elif style == "intl_zero":
        text = f"0044 7700 900{last3:03d}"
    elif style == "brackets":
        text = f"(07700) 900{last3:03d}"
    else:
        text = f"07700-900-{last3:03d}"
    return text, e164


def us_ca_phone(rng: random.Random, country: str) -> tuple[str, str]:
    """Return (display text, E.164) for a fictional US/CA number in the 555-01xx range."""
    area = rng.choice(US_AREA_CODES if country == "US" else CA_AREA_CODES)
    last2 = rng.randint(0, 99)
    line = f"01{last2:02d}"
    e164 = f"+1{area}555{line}"
    style = rng.choice(["parens_dash", "dashed", "intl_spaced", "intl_dashed", "dotted"])
    if style == "parens_dash":
        text = f"({area}) 555-{line}"
    elif style == "dashed":
        text = f"{area}-555-{line}"
    elif style == "intl_spaced":
        text = f"+1 {area} 555 {line}"
    elif style == "intl_dashed":
        text = f"1-{area}-555-{line}"
    else:
        text = f"{area}.555.{line}"
    return text, e164


def phone_for_country(rng: random.Random, country: str) -> tuple[str, str]:
    return uk_phone(rng) if country == "GB" else us_ca_phone(rng, country)


def format_amount(rng: random.Random, amount: float, currency: str) -> str:
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}[currency]
    style = rng.choice(["code_comma", "symbol_k", "symbol_decimal", "symbol_comma"])
    if amount >= 1000 and amount % 1000 == 0 and style == "symbol_k":
        return f"{symbol}{int(amount // 1000)}k"
    if style == "code_comma":
        return f"{currency} {amount:,.0f}"
    if style == "symbol_decimal":
        return f"{symbol}{amount:,.2f}"
    return f"{symbol}{amount:,.0f}"


def make_company_name(rng: random.Random, sector: str, country: str) -> str:
    prefix = rng.choice(COMPANY_PREFIXES)
    suffix = rng.choice(SECTOR_SUFFIXES[sector])
    form = rng.choice(LEGAL_FORMS[country])
    name = f"{prefix} {suffix} {form}"
    check_no_blocklisted_brand(name)
    return name


def slugify(text: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    return re.sub(r"-+", "-", raw).strip("-")


def make_person_name(rng: random.Random) -> tuple[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    check_no_blocklisted_brand(f"{first} {last}")
    return first, last


def postcode_for(rng: random.Random, country: str) -> str:
    if country == "GB":
        area = rng.choice(["BS", "M", "LS", "EC", "SW", "B", "CF", "G"])
        gb_letters = "ABDEFGHJLNPQRSTUWXYZ"
        return f"{area}{rng.randint(1, 20)} {rng.randint(1, 9)}{rng.choice(gb_letters)}{rng.choice(gb_letters)}"
    if country == "US":
        return f"{rng.randint(10000, 99999)}"
    letters = "ABCEGHJKLMNPRSTVXY"
    return (
        f"{rng.choice(letters)}{rng.randint(0, 9)}{rng.choice(letters)} "
        f"{rng.randint(0, 9)}{rng.choice(letters)}{rng.randint(0, 9)}"
    )


def address_line(rng: random.Random, country: str) -> tuple[str, str]:
    """Return (street line, city) for the given country."""
    number = rng.randint(1, 220)
    street = rng.choice(STREETS)
    city = rng.choice(CITIES[country])
    return f"{number} {street}", city


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------

EXTRACTION_TEMPLATES = [
    "We came across {company} while comparing options for {product} and would like to learn more.",
    "Our team at {company} is evaluating {product} for a rollout across the {sector} side of the business.",
    "I'm reaching out for {company}; we need a replacement for our current tooling and {product} looks promising.",
    "{company} is expanding and we're now looking at {product} to support the growth.",
    "Following a recommendation from a partner, {company} would like a demo of {product}.",
    "We're renewing our stack this quarter and {company} wants to shortlist {product}.",
]
EXTRACTION_SEATS_TEMPLATES = [
    "We'd be rolling this out to a team of {seats} people.",
    "The initial deployment would cover {seats} seats, with room to grow.",
    "Roughly {seats} of us would need access from day one.",
]
EXTRACTION_BUDGET_TEMPLATES = [
    "Our budget for this is around {amount}.",
    "We've set aside {amount} for the first year.",
    "Finance approved a ceiling of {amount} for this project.",
]
EXTRACTION_MEETING_TEMPLATES = [
    "Could we set up a call {phrase}?",
    "We'd like to talk this through {phrase} if that works for you.",
    "Are you available for a short demo {phrase}?",
]
EXTRACTION_URGENCY_HINTS = {
    "high": [
        "This is time-sensitive; our current contract lapses very soon.",
        "We need to move quickly on this, ideally within the week.",
    ],
    "low": [
        "There's no rush on our end, we're just gathering information for now.",
        "This is an early-stage enquiry, no immediate timeline.",
    ],
    "normal": [
        "We're hoping to make a decision in the next month or so.",
    ],
}
COUNTRY_MENTION_STYLES = ["name", "code", "implied"]
CLOSINGS = [
    "Looking forward to hearing from you.",
    "Thanks in advance for your time.",
    "Let me know what the next steps would be.",
    "Happy to answer any questions on our side.",
]


def build_extraction_item(rng: random.Random, item_id: str) -> dict[str, Any]:
    country = rng.choice(["GB", "US", "CA"])
    sector = rng.choice(SECTORS)
    company = make_company_name(rng, sector, country)
    first, last = make_person_name(rng)
    contact_name = f"{first} {last}"
    domain_slug = slugify(company.replace(" Ltd", "").replace(" Inc.", "").replace(" LLC", "").replace(" LLP", ""))
    tld = rng.choice(["com", "org", "net"])
    contact_email = f"{first[0].lower()}.{last.lower()}@{domain_slug}.example.{tld}"
    phone_text, phone_e164 = phone_for_country(rng, country)
    product = rng.choice(PRODUCT_LINES)

    ref_year = 2026
    ref_date = date(ref_year, rng.randint(1, 12), rng.randint(1, 28))

    seats = rng.choice([None, rng.randint(5, 400)])
    if rng.random() < 0.55:
        currency = rng.choice(["EUR", "USD", "GBP"])
        amount = float(rng.choice([2500, 4200, 8000, 12500, 15000, 25000, 42000, 60000]))
    else:
        currency = None
        amount = None
    phrase, meeting_iso = meeting_date_expression(rng, ref_date)
    urgency = rng.choices(["low", "normal", "high"], weights=[0.25, 0.5, 0.25])[0]

    street, city = address_line(rng, country)
    country_style = rng.choice(COUNTRY_MENTION_STYLES)
    if country_style == "name":
        addr_country = COUNTRY_NAMES[country]
    elif country_style == "code":
        addr_country = country
    else:
        addr_country = ""

    body_lines = [
        f"From: {contact_name} <{contact_email}>",
        f"Subject: Enquiry about {product}",
        "",
        "Hi there,",
        "",
        rng.choice(EXTRACTION_TEMPLATES).format(company=company, product=product, sector=sector),
    ]
    if seats:
        body_lines.append(rng.choice(EXTRACTION_SEATS_TEMPLATES).format(seats=seats))
    if amount is not None and currency is not None:
        amount_text = format_amount(rng, amount, currency)
        body_lines.append(rng.choice(EXTRACTION_BUDGET_TEMPLATES).format(amount=amount_text))
    if phrase:
        body_lines.append(rng.choice(EXTRACTION_MEETING_TEMPLATES).format(phrase=phrase))
    body_lines.append(rng.choice(EXTRACTION_URGENCY_HINTS[urgency]))

    addr_bits = [street, city]
    if addr_country:
        addr_bits.append(addr_country)
    body_lines.append(f"You can reach us at {street}, {city}" + (f", {addr_country}." if addr_country else "."))
    body_lines.append(f"Phone: {phone_text}")
    body_lines.append("")
    body_lines.append(rng.choice(CLOSINGS))
    body_lines.append("")
    body_lines.append(contact_name)
    body_lines.append(company)

    text = f"Date: {format_date_header(ref_date)}\n" + "\n".join(body_lines)

    gold = {
        "company_name": company,
        "contact_name": contact_name,
        "contact_email": contact_email,
        "phone": phone_e164,
        "country": country,
        "product": product,
        "seats": seats,
        "budget_amount": amount,
        "budget_currency": currency,
        "requested_meeting_date": meeting_iso,
        "urgency": urgency,
    }
    return {"id": item_id, "text": text, "reference_date": iso(ref_date), "gold": gold}


def build_extraction(seed: int, out_dir: Path) -> DatasetCard:
    rng = random.Random(f"extraction:{seed}")
    calib = [build_extraction_item(rng, f"extraction-calib-{i:04d}") for i in range(100)]
    test = [build_extraction_item(rng, f"extraction-test-{i:04d}") for i in range(200)]

    task_dir = out_dir / "extraction"
    write_jsonl(task_dir / "calib.jsonl", calib)
    write_jsonl(task_dir / "test.jsonl", test)

    fields = {
        "company_name": FieldSpec(type="string", required=True, grounded=True, nullable=False),
        "contact_name": FieldSpec(type="string", required=True, grounded=True, nullable=False),
        "contact_email": FieldSpec(type="email", required=True, grounded=True, nullable=False),
        "phone": FieldSpec(type="phone", required=True, nullable=False),
        "country": FieldSpec(type="country", required=True, nullable=False),
        "product": FieldSpec(type="enum", values=PRODUCT_LINES, required=True, nullable=False),
        "seats": FieldSpec(type="int", required=False, nullable=True),
        "budget_amount": FieldSpec(type="amount", required=False, nullable=True),
        "budget_currency": FieldSpec(type="currency", values=["EUR", "USD", "GBP"], required=False, nullable=True),
        "requested_meeting_date": FieldSpec(type="date", required=False, nullable=True),
        "urgency": FieldSpec(type="enum", values=["low", "normal", "high"], required=True, nullable=False),
    }
    card = DatasetCard(
        source="synthetic",
        licence="Apache-2.0",
        seed=seed,
        metric="field_accuracy",
        distinct_templates=(
            len(EXTRACTION_TEMPLATES)
            + len(EXTRACTION_SEATS_TEMPLATES)
            + len(EXTRACTION_BUDGET_TEMPLATES)
            + len(EXTRACTION_MEETING_TEMPLATES)
            + sum(len(v) for v in EXTRACTION_URGENCY_HINTS.values())
            + len(CLOSINGS)
        ),
        notes="Inbound B2B enquiry emails; gold fields resolved by the generator itself.",
    )
    spec = TaskSpec(
        name="extraction",
        kind="extraction",
        description="Inbound B2B enquiry email to CRM fields.",
        calib="calib.jsonl",
        test="test.jsonl",
        fields=fields,
        card=card,
    )
    write_yaml_task(task_dir / "task.yaml", spec)
    return card


# --------------------------------------------------------------------------------------
# pii_redaction
# --------------------------------------------------------------------------------------

FULL_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
PII_TYPES = ["PERSON", "EMAIL", "PHONE", "IBAN", "ADDRESS", "DATE_OF_BIRTH"]
PII_CATEGORIES = ["support_ticket", "hr_note", "invoice_dispute", "call_note"]

CATEGORY_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "support_ticket": {
        "PERSON": 0.25,
        "EMAIL": 0.25,
        "PHONE": 0.20,
        "ADDRESS": 0.15,
        "IBAN": 0.05,
        "DATE_OF_BIRTH": 0.10,
    },
    "hr_note": {
        "PERSON": 0.30,
        "DATE_OF_BIRTH": 0.25,
        "ADDRESS": 0.20,
        "PHONE": 0.10,
        "EMAIL": 0.10,
        "IBAN": 0.05,
    },
    "invoice_dispute": {
        "PERSON": 0.20,
        "EMAIL": 0.15,
        "PHONE": 0.10,
        "IBAN": 0.35,
        "ADDRESS": 0.15,
        "DATE_OF_BIRTH": 0.05,
    },
    "call_note": {
        "PERSON": 0.35,
        "PHONE": 0.35,
        "EMAIL": 0.15,
        "ADDRESS": 0.10,
        "IBAN": 0.03,
        "DATE_OF_BIRTH": 0.02,
    },
}

SUPPORT_ISSUES = [
    "Delayed delivery",
    "Damaged item on arrival",
    "Login access issue",
    "Billing discrepancy",
    "Missing invoice",
    "Product setup help",
]
HR_TOPICS = [
    "Onboarding checklist",
    "Leave request follow-up",
    "Benefits enrolment",
    "Reference check",
    "Payroll correction",
]
CALL_TOPICS = [
    "Inbound enquiry",
    "Follow-up call",
    "Renewal discussion",
    "Complaint handling",
    "Technical support call",
]
PII_OPENINGS = {
    "support_ticket": lambda rng: f"Subject: {rng.choice(SUPPORT_ISSUES)}\nTicket #{rng.randint(10000, 99999)}\n\n",
    "hr_note": lambda rng: f"HR file note - {rng.choice(HR_TOPICS)}\n\n",
    "invoice_dispute": lambda rng: f"Invoice dispute - INV-{rng.randint(2000, 9999)}\n\n",
    "call_note": lambda rng: f"Call notes - {rng.choice(CALL_TOPICS)}\n\n",
}
PII_FILLERS: dict[str, list[str]] = {
    "support_ticket": [
        "Customer says the issue started last week.",
        "Priority has been set to medium.",
        "Awaiting customer confirmation before closing.",
        "Escalated to tier two support.",
    ],
    "hr_note": [
        "Documented for the personnel file.",
        "No further action required at this time.",
        "Manager has been informed.",
        "Follow-up scheduled for next review cycle.",
    ],
    "invoice_dispute": [
        "Amount in dispute is under review.",
        "Finance team has been copied.",
        "Customer disputes the line item total.",
        "Credit note may be required.",
    ],
    "call_note": [
        "Call lasted about ten minutes.",
        "Customer sounded satisfied with the outcome.",
        "No further callback requested.",
        "Logged for quality assurance.",
    ],
}

TYPE_TEMPLATES: dict[str, list[str]] = {
    "PERSON": [
        "Reported by {v}.",
        "{v} called in about this.",
        "Please loop in {v} from the account team.",
        "Contact: {v}.",
    ],
    "EMAIL": [
        "You can reach the customer at {v}.",
        "Please cc {v} on all correspondence.",
        "Confirmation was sent to {v}.",
        "Reply-to address on file: {v}.",
    ],
    "PHONE": [
        "Callback number: {v}.",
        "The customer's direct line is {v}.",
        "We tried reaching {v} twice.",
        "Preferred contact number: {v}.",
    ],
    "IBAN": [
        "Refund account: {v}.",
        "Please process the reimbursement to {v}.",
        "Bank details on file: {v}.",
        "Payment should be returned to {v}.",
    ],
    "ADDRESS": [
        "Shipping address: {v}.",
        "The site visit is scheduled at {v}.",
        "Correspondence address: {v}.",
        "Please update our records to {v}.",
    ],
    "DATE_OF_BIRTH": [
        "Date of birth: {v}.",
        "Identity verified against DOB {v}.",
        "Born on {v} per the HR file.",
        "DOB on record: {v}.",
    ],
}
HARD_NEGATIVE_TEMPLATES: dict[str, list[str]] = {
    "company": [
        "This relates to our contract with {v}.",
        "The escalation was raised by {v}.",
        "{v} is the account in question.",
    ],
    "product": ["The issue concerns the {v} module.", "This ticket is about {v}.", "{v} usage triggered the alert."],
    "order_id": ["Reference: {v}.", "Related order: {v}.", "See also {v}."],
    "non_birth_date": ["Ticket opened on {v}.", "Last contacted on {v}.", "Renewal is due {v}."],
    "switchboard": ["Company switchboard: {v}.", "Main office line: {v}.", "General enquiries: {v}."],
}


def format_date_of_birth(rng: random.Random, d: date) -> str:
    style = rng.choice(["dmy_words", "dmy_slash", "mdy_words"])
    if style == "dmy_words":
        return f"{d.day} {FULL_MONTHS[d.month - 1]} {d.year}"
    if style == "dmy_slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    return f"{FULL_MONTHS[d.month - 1]} {d.day}, {d.year}"


IBAN_STRUCTURE: dict[str, list[tuple[int, str]]] = {
    "GB": [(4, "alpha"), (14, "digit")],
    "IE": [(4, "alpha"), (14, "digit")],
    "DE": [(18, "digit")],
    "FR": [(23, "digit")],
    "NL": [(4, "alpha"), (10, "digit")],
    "ES": [(20, "digit")],
}


def _iban_check_digits(country: str, bban: str) -> str:
    rearranged = bban + country + "00"
    numeric = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    remainder = int(numeric) % 97
    return f"{98 - remainder:02d}"


def generate_iban(rng: random.Random) -> str:
    country = rng.choice(list(IBAN_STRUCTURE))
    bban = ""
    for length, kind in IBAN_STRUCTURE[country]:
        if kind == "alpha":
            bban += "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))
        else:
            bban += "".join(str(rng.randint(0, 9)) for _ in range(length))
    return f"{country}{_iban_check_digits(country, bban)}{bban}"


def pii_person_value(rng: random.Random) -> str:
    first, last = make_person_name(rng)
    return f"{first} {last}"


def pii_email_value(rng: random.Random) -> str:
    first, last = make_person_name(rng)
    tld = rng.choice(["com", "org", "net"])
    sub = rng.choice(["", "webmail.", "mail."])
    return f"{first.lower()}.{last.lower()}@{sub}example.{tld}"


def pii_phone_value(rng: random.Random) -> str:
    return phone_for_country(rng, rng.choice(["GB", "US", "CA"]))[0]


def pii_address_value(rng: random.Random) -> str:
    country = rng.choice(["GB", "US", "CA"])
    street, city = address_line(rng, country)
    return f"{street}, {city}, {postcode_for(rng, country)}"


def pii_dob_value(rng: random.Random) -> str:
    d = date(rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28))
    return format_date_of_birth(rng, d)


PII_VALUE_FN = {
    "PERSON": pii_person_value,
    "EMAIL": pii_email_value,
    "PHONE": pii_phone_value,
    "IBAN": lambda rng: generate_iban(rng),
    "ADDRESS": pii_address_value,
    "DATE_OF_BIRTH": pii_dob_value,
}
HARD_NEGATIVE_VALUE_FN = {
    "company": lambda rng: make_company_name(rng, rng.choice(SECTORS), rng.choice(["GB", "US", "CA"])),
    "product": lambda rng: rng.choice(PRODUCT_LINES),
    "order_id": lambda rng: rng.choice([f"ORD-{rng.randint(10000, 99999)}", f"INV-{rng.randint(1000, 9999)}"]),
    "non_birth_date": lambda rng: format_date_of_birth(
        rng, date(rng.randint(2024, 2026), rng.randint(1, 12), rng.randint(1, 28))
    ),
    "switchboard": lambda rng: phone_for_country(rng, rng.choice(["GB", "US", "CA"]))[0],
}


class DocBuilder:
    """Assembles a document while tracking exact char offsets for embedded gold spans."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._spans: list[dict[str, Any]] = []

    def _pos(self) -> int:
        return sum(len(s) for s in self._buf)

    def write(self, text: str, type_: str | None = None) -> None:
        start = self._pos()
        self._buf.append(text)
        if type_ is not None:
            self._spans.append({"start": start, "end": start + len(text), "type": type_, "value": text})

    def build(self) -> tuple[str, list[dict[str, Any]]]:
        text = "".join(self._buf).rstrip()
        spans = []
        for s in sorted(self._spans, key=lambda s: s["start"]):
            assert text[s["start"] : s["end"]] == s["value"], f"span offset mismatch: {s}"
            spans.append({"start": s["start"], "end": s["end"], "type": s["type"]})
        return text, spans


def write_templated(db: DocBuilder, template: str, value: str, type_: str | None) -> None:
    pre, _, post = template.partition("{v}")
    db.write(pre)
    db.write(value, type_)
    db.write(post)


def _pii_free_flags(rng: random.Random, n: int, share: float = 0.15) -> list[bool]:
    """Exactly round(share * n) PII-free documents per split, at seeded positions."""
    flags = [i < round(share * n) for i in range(n)]
    rng.shuffle(flags)
    return flags


def build_pii_item(rng: random.Random, item_id: str, no_pii: bool) -> dict[str, Any]:
    category = rng.choice(PII_CATEGORIES)
    n_spans = 0 if no_pii else rng.choices([1, 2, 3, 4, 5, 6], weights=[30, 25, 20, 13, 8, 4])[0]
    weights = CATEGORY_TYPE_WEIGHTS[category]
    gold_types = rng.choices(list(weights), weights=list(weights.values()), k=n_spans)

    db = DocBuilder()
    db.write(PII_OPENINGS[category](rng))

    actions: list[Any] = []
    for t in gold_types:
        value = PII_VALUE_FN[t](rng)
        tmpl = rng.choice(TYPE_TEMPLATES[t])
        actions.append((tmpl, value, t))

    hard_kinds = rng.sample(list(HARD_NEGATIVE_VALUE_FN), k=rng.randint(1, 3))
    for hk in hard_kinds:
        value = HARD_NEGATIVE_VALUE_FN[hk](rng)
        tmpl = rng.choice(HARD_NEGATIVE_TEMPLATES[hk])
        actions.append((tmpl, value, None))

    fillers = [(filler, None, None) for filler in rng.sample(PII_FILLERS[category], k=rng.randint(1, 2))]

    ordered: list[tuple[str, str | None, str | None]] = actions + fillers
    rng.shuffle(ordered)
    for tmpl, value, type_ in ordered:
        if value is None:
            db.write(tmpl)
        else:
            write_templated(db, tmpl, value, type_)
        db.write(" ")

    text, spans = db.build()
    return {"id": item_id, "text": text, "spans": spans}


def build_pii(seed: int, out_dir: Path) -> DatasetCard:
    rng = random.Random(f"pii_redaction:{seed}")
    calib_free = _pii_free_flags(rng, 100)
    test_free = _pii_free_flags(rng, 200)
    calib = [build_pii_item(rng, f"pii-calib-{i:04d}", calib_free[i]) for i in range(100)]
    test = [build_pii_item(rng, f"pii-test-{i:04d}", test_free[i]) for i in range(200)]

    task_dir = out_dir / "pii_redaction"
    write_jsonl(task_dir / "calib.jsonl", calib)
    write_jsonl(task_dir / "test.jsonl", test)

    distinct_templates = (
        sum(len(v) for v in TYPE_TEMPLATES.values())
        + sum(len(v) for v in HARD_NEGATIVE_TEMPLATES.values())
        + sum(len(v) for v in PII_FILLERS.values())
        + len(SUPPORT_ISSUES)
        + len(HR_TOPICS)
        + len(CALL_TOPICS)
    )
    card = DatasetCard(
        source="synthetic",
        licence="Apache-2.0",
        seed=seed,
        metric="f2",
        distinct_templates=distinct_templates,
        notes="Support tickets, HR notes, invoice disputes and call notes with 0-6 PII spans; ~15% PII-free.",
    )
    spec = TaskSpec(
        name="pii_redaction",
        kind="pii_redaction",
        description="Redact personal data from business notes.",
        calib="calib.jsonl",
        test="test.jsonl",
        pii_types=PII_TYPES,
        card=card,
    )
    write_yaml_task(task_dir / "task.yaml", spec)
    return card


# --------------------------------------------------------------------------------------
# summarisation (+ judge-calib / judge-holdout)
# --------------------------------------------------------------------------------------

MEETING_TOPICS = [
    "the website migration",
    "the vendor consolidation",
    "the office relocation",
    "the Q4 hiring plan",
    "the customer onboarding redesign",
    "the warehouse automation rollout",
    "the support ticket backlog",
    "the annual budget review",
    "the new supplier onboarding",
    "the security audit follow-up",
]
WORKSTREAMS = [
    "the vendor contract",
    "the rollout schedule",
    "the support queue",
    "the integration testing",
    "the training materials",
    "the compliance review",
    "the budget forecast",
    "the staffing plan",
    "the migration plan",
    "the customer feedback",
    "the pilot results",
    "the security review",
    "the deployment checklist",
    "the onboarding flow",
    "the incident backlog",
    "the license renewal",
    "the hardware refresh",
    "the vendor scorecard",
    "the escalation process",
    "the reporting dashboard",
    "the change request queue",
    "the capacity plan",
    "the risk register",
    "the audit findings",
    "the service catalogue",
    "the automation backlog",
    "the network upgrade",
    "the data migration",
    "the access review",
    "the disaster-recovery plan",
]
DUE_DATES = [
    "Friday",
    "next Wednesday",
    "the end of the month",
    "14 April",
    "the 20th",
    "next Monday",
    "the end of the quarter",
    "Thursday",
    "the 3rd",
    "next Friday",
]
AMOUNT_VALUES = [3500, 6000, 8500, 12000, 18000, 24000, 32000, 50000, 9500, 15500]
AMOUNT_SYMBOLS = ["$", "£", "€"]
DECISION_ACTIONS = [
    "move forward with the new vendor",
    "extend the pilot to all regions",
    "postpone the launch",
    "consolidate the two systems",
    "adopt the revised timeline",
    "proceed with the in-house option",
    "finalise the contract terms",
    "roll out the update company-wide",
    "freeze scope for this quarter",
    "switch to the new reporting tool",
]
TASK_ACTIONS = [
    "send the updated proposal",
    "follow up with the vendor",
    "prepare the migration checklist",
    "schedule the kickoff session",
    "draft the revised budget",
    "coordinate with the facilities team",
    "circulate the meeting notes",
    "confirm the go-live date",
    "update the project tracker",
    "arrange the site visit",
]
ALT_APPROACHES = [
    "a phased rollout",
    "an in-house build",
    "the previous vendor",
    "a manual process",
    "outsourcing the work",
    "a smaller pilot",
    "the legacy system",
    "a same-day migration",
]
OPENING_TURNS = [
    "{speaker}: Thanks everyone for joining, let's get started on {topic}.",
    "{speaker}: Good morning all, let's dive straight into {topic}.",
    "{speaker}: Quick reminder that we're on a tight schedule today, so let's cover {topic} first.",
    "{speaker}: Appreciate everyone making time, today is mostly about {topic}.",
]
CLOSING_TURNS = [
    "{speaker}: Great, thanks everyone, I'll circulate notes by end of day.",
    "{speaker}: Sounds good, same time next week?",
    "{speaker}: Appreciate everyone's time, that's a wrap.",
    "{speaker}: I'll follow up over email with any loose ends.",
]
SMALL_TALK = [
    "{speaker}: Sorry I'm a few minutes late, previous call ran over.",
    "{speaker}: Can everyone hear me okay?",
    "{speaker}: I'll share my screen in a second.",
    "{speaker}: Before we dive in, congrats on the milestone last week.",
    "{speaker}: Let's take five minutes at the end for questions.",
    "{speaker}: I grabbed coffee on the way, hope that's fine.",
    "{speaker}: Glad we're not commuting today, the weather's been rough.",
    "{speaker}: Let's keep this to thirty minutes if we can.",
    "{speaker}: I'll send the notes around afterwards.",
    "{speaker}: Can we push the deep dive to next week's sync?",
    "{speaker}: Mind if I join from my phone, I'm between rooms.",
    "{speaker}: Let's mute when we're not talking, there's some echo.",
]
SUPERSEDED = [
    "{speaker}: We originally looked at {alt}, but that's now off the table.",
    "{speaker}: The earlier plan involving {alt} didn't pan out, so we moved on.",
    "{speaker}: We'd considered {alt} last quarter, but priorities shifted.",
    "{speaker}: That approach around {alt} was shelved after the last review.",
]
REJECTED = [
    "{speaker}: We looked at {alt} and decided against it, cost was too high.",
    "{speaker}: {alt} was on the shortlist but didn't meet the requirements.",
    "{speaker}: We ruled out {alt} early on.",
    "{speaker}: The option of {alt} couldn't meet our timeline, so we passed.",
]
ELABORATIONS = [
    "{speaker}: On {ws}, we're roughly {pct}% through and tracking close to plan.",
    "{speaker}: The main risk on {ws} is still {reason}, but there's a mitigation in place.",
    "{speaker}: We had a good conversation with stakeholders about {ws} earlier this week.",
    "{speaker}: {ws_cap} is taking a bit longer than expected, mostly due to {reason}.",
    "{speaker}: I think {ws} is in decent shape, just a few loose ends to tie up.",
    "{speaker}: There's some dependency between {ws} and the wider rollout worth flagging.",
    "{speaker}: Feedback on {ws} has been mostly positive so far.",
    "{speaker}: We'll need another sync specifically on {ws} before we can close it out.",
    "{speaker}: {ws_cap} came up a few times in customer conversations this month.",
    "{speaker}: Let's make sure {ws} doesn't slip, it's on the critical path.",
    "{speaker}: I don't have much to add on {ws}, it's progressing as expected.",
    "{speaker}: The numbers on {ws} look reasonable, nothing alarming there.",
    "{speaker}: We should loop in another team on {ws} at some point.",
    "{speaker}: {ws_cap} is mostly a resourcing question at this stage.",
    "{speaker}: Once {ws} wraps up we can shift focus to the next item.",
    "{speaker}: There were a couple of questions on {ws} in the last review.",
    "{speaker}: {ws_cap} has been the smoothest part of this whole effort, honestly.",
    "{speaker}: We're keeping a close eye on {ws} given how visible it is.",
    "{speaker}: A quick update on {ws}: still on track, no blockers right now.",
    "{speaker}: {ws_cap} might need a bit more budget, we'll confirm next week.",
    "{speaker}: Worth noting that {ws} depends on sign-off from another team.",
    "{speaker}: Nothing new on {ws} since last time, just steady progress.",
    "{speaker}: We're a little behind on {ws}, but nothing we can't recover from.",
    "{speaker}: {ws_cap} is ahead of schedule, which frees up time for the rest.",
    "{speaker}: The team working on {ws} could use another pair of hands.",
]
REASON_WORDS = [
    "a staffing gap",
    "a vendor delay",
    "a scope change",
    "a dependency on another team",
    "a data-quality issue",
    "a permissions issue",
    "an unexpected outage",
    "a licensing hold-up",
]

SUMMARY_STYLES: dict[str, list[str]] = {
    "decision": [
        "The team decided to {action} by {date}.",
        "A decision was made to {action}, targeting {date}.",
        "Decision: {action}, by {date}.",
    ],
    "action_item": [
        "{owner} will {action} by {date}.",
        "It was agreed that {owner} would {action} by {date}.",
        "Action: {owner} to {action}, due {date}.",
    ],
    "amount": [
        "Budget for {ws} was set at {amount}.",
        "The team agreed to allocate {amount} for {ws}.",
        "Budget: {amount} for {ws}.",
    ],
}


def _amount_text(rng: random.Random, value: int) -> str:
    return f"{rng.choice(AMOUNT_SYMBOLS)}{value:,}"


def build_required_facts(rng: random.Random, speakers: list[str], n_facts: int) -> list[dict[str, Any]]:
    kinds = ["decision", "action_item", "amount"]
    plan = kinds[:] + rng.choices(kinds, k=n_facts - len(kinds))
    rng.shuffle(plan)
    facts = []
    for idx, kind in enumerate(plan):
        speaker = rng.choice(speakers)
        if kind == "decision":
            action = rng.choice(DECISION_ACTIONS)
            due = rng.choice(DUE_DATES)
            turn = f"{speaker}: We've decided to {action} by {due}."
            fact_text = f"The team decided to {action} by {due}."
            key_tokens = {"date": due}
            internal = {"action": action, "date": due, "ws": None, "amount": None, "owner": None}
        elif kind == "action_item":
            owner = rng.choice(speakers)
            action = rng.choice(TASK_ACTIONS)
            due = rng.choice(DUE_DATES)
            turn = f"{speaker}: {owner} will {action} by {due}."
            fact_text = f"{owner} will {action} by {due}."
            key_tokens = {"owner": owner, "date": due}
            internal = {"action": action, "date": due, "ws": None, "amount": None, "owner": owner}
        else:
            ws = rng.choice(WORKSTREAMS)
            value = rng.choice(AMOUNT_VALUES)
            amount_text = _amount_text(rng, value)
            turn = f"{speaker}: The budget for {ws} is set at {amount_text}."
            fact_text = f"Budget for {ws} was set at {amount_text}."
            key_tokens = {"amount": amount_text}
            internal = {"action": None, "date": None, "ws": ws, "amount": amount_text, "owner": None}
        facts.append(
            {
                "index": idx,
                "kind": kind,
                "fact": fact_text,
                "key_tokens": key_tokens,
                "turn": turn,
                "_internal": internal,
            }
        )
    return facts


def public_fact(f: dict[str, Any]) -> dict[str, Any]:
    return {"index": f["index"], "kind": f["kind"], "fact": f["fact"], "key_tokens": f["key_tokens"]}


def build_transcript(rng: random.Random, item_id: str) -> dict[str, Any]:
    speakers = rng.sample(FIRST_NAMES, k=rng.choice([3, 4]))
    topic = rng.choice(MEETING_TOPICS)
    workstreams = rng.sample(WORKSTREAMS, k=4)
    n_facts = rng.choice([4, 5, 6])
    n_distractors = rng.randint(5, 10)

    facts = build_required_facts(rng, speakers, n_facts)

    turns: list[str] = []
    for f in facts:
        turns.append(f["turn"])

    kind_pool = ["small_talk"] * 5 + ["superseded"] * 3 + ["rejected"] * 3
    for _ in range(n_distractors):
        speaker = rng.choice(speakers)
        kind = rng.choice(kind_pool)
        if kind == "small_talk":
            turns.append(rng.choice(SMALL_TALK).format(speaker=speaker))
        elif kind == "superseded":
            turns.append(rng.choice(SUPERSEDED).format(speaker=speaker, alt=rng.choice(ALT_APPROACHES)))
        else:
            turns.append(rng.choice(REJECTED).format(speaker=speaker, alt=rng.choice(ALT_APPROACHES)))

    rng.shuffle(turns)
    opening = rng.choice(OPENING_TURNS).format(speaker=rng.choice(speakers), topic=topic)
    closing = rng.choice(CLOSING_TURNS).format(speaker=rng.choice(speakers))
    all_turns = [opening, *turns, closing]

    def word_count(ts: list[str]) -> int:
        return sum(len(t.split()) for t in ts)

    target = rng.randint(650, 950)
    guard = 0
    while word_count(all_turns) < target and guard < 200:
        guard += 1
        speaker = rng.choice(speakers)
        ws = rng.choice(workstreams)
        tmpl = rng.choice(ELABORATIONS)
        pct = rng.choice([15, 20, 30, 40, 55, 60, 70, 80, 90])
        line = tmpl.format(
            speaker=speaker, ws=ws, ws_cap=ws[0].upper() + ws[1:], pct=pct, reason=rng.choice(REASON_WORDS)
        )
        all_turns.insert(rng.randint(1, len(all_turns) - 1), line)

    text = "\n".join(all_turns)
    return {
        "id": item_id,
        "text": text,
        "max_words": 120,
        "required_facts": [public_fact(f) for f in facts],
        "_facts": facts,
    }


def render_fact_sentence(
    f: dict[str, Any], style: int, mutate: str | None = None, mutated_value: str | None = None
) -> str:
    internal = f["_internal"]
    tmpl = SUMMARY_STYLES[f["kind"]][style]
    slots = dict(internal)
    if mutate and mutated_value is not None:
        slots[mutate] = mutated_value
    return tmpl.format(
        action=slots.get("action"),
        date=slots.get("date"),
        owner=slots.get("owner"),
        ws=slots.get("ws"),
        amount=slots.get("amount"),
    )


def build_judge_summaries(
    rng: random.Random, transcript: dict[str, Any], style_counter: list[int], id_prefix: str
) -> list[dict[str, Any]]:
    facts = transcript["_facts"]
    n = len(facts)
    records = []
    variants = ["faithful", "fact_dropped", "wrong_number_or_date", "wrong_owner", "invented_commitment"]
    for variant in variants:
        style = style_counter[0] % 3
        style_counter[0] += 1
        sentences = []
        labels = []
        unsupported = 0
        if variant == "faithful":
            for f in facts:
                sentences.append(render_fact_sentence(f, style))
                labels.append({"index": f["index"], "present": True, "correct": True})
        elif variant == "fact_dropped":
            drop_idx = rng.randrange(n)
            for f in facts:
                if f["index"] == drop_idx:
                    labels.append({"index": f["index"], "present": False, "correct": False})
                    continue
                sentences.append(render_fact_sentence(f, style))
                labels.append({"index": f["index"], "present": True, "correct": True})
        elif variant == "wrong_number_or_date":
            target_idx = rng.randrange(n)
            for f in facts:
                if f["index"] == target_idx:
                    if f["kind"] == "amount":
                        current = int(re.sub(r"[^0-9]", "", f["_internal"]["amount"]))
                        wrong = _amount_text(rng, rng.choice([v for v in AMOUNT_VALUES if v != current]))
                        sentences.append(render_fact_sentence(f, style, mutate="amount", mutated_value=wrong))
                    else:
                        wrong = rng.choice([d for d in DUE_DATES if d != f["_internal"]["date"]])
                        sentences.append(render_fact_sentence(f, style, mutate="date", mutated_value=wrong))
                    labels.append({"index": f["index"], "present": True, "correct": False})
                else:
                    sentences.append(render_fact_sentence(f, style))
                    labels.append({"index": f["index"], "present": True, "correct": True})
        elif variant == "wrong_owner":
            owner_facts = [f for f in facts if f["kind"] == "action_item"]
            target = rng.choice(owner_facts)
            for f in facts:
                if f["index"] == target["index"]:
                    wrong = rng.choice([n for n in FIRST_NAMES if n != f["_internal"]["owner"]])
                    sentences.append(render_fact_sentence(f, style, mutate="owner", mutated_value=wrong))
                    labels.append({"index": f["index"], "present": True, "correct": False})
                else:
                    sentences.append(render_fact_sentence(f, style))
                    labels.append({"index": f["index"], "present": True, "correct": True})
        else:  # invented_commitment
            for f in facts:
                sentences.append(render_fact_sentence(f, style))
                labels.append({"index": f["index"], "present": True, "correct": True})
            owner = rng.choice(FIRST_NAMES)
            action = rng.choice(TASK_ACTIONS)
            due = rng.choice(DUE_DATES)
            sentences.append(f"{owner} also committed to {action} by {due}.")
            unsupported = 1

        summary = " ".join(sentences)
        wc = len(summary.split())
        assert wc <= 120, f"summary too long ({wc} words): {summary!r}"
        present_correct = sum(1 for label in labels if label["present"] and label["correct"])
        label_pass = present_correct >= math.ceil(0.8 * n) and unsupported == 0 and wc <= 120
        records.append(
            {
                "id": f"{id_prefix}-{transcript['id']}-{variant}",
                "transcript_id": transcript["id"],
                "variant": variant,
                "style": style,
                "summary": summary,
                "facts": labels,
                "unsupported_claims": unsupported,
                "label_pass": label_pass,
            }
        )
    return records


def build_summarisation(seed: int, out_dir: Path) -> DatasetCard:
    rng = random.Random(f"summarisation:{seed}")
    calib = [build_transcript(rng, f"summarisation-calib-{i:04d}") for i in range(40)]
    test = [build_transcript(rng, f"summarisation-test-{i:04d}") for i in range(80)]
    holdout = [build_transcript(rng, f"summarisation-holdout-{i:04d}") for i in range(40)]

    task_dir = out_dir / "summarisation"

    def strip_internal(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in it.items() if k != "_facts"} for it in items]

    write_jsonl(task_dir / "calib.jsonl", strip_internal(calib))
    write_jsonl(task_dir / "test.jsonl", strip_internal(test))
    write_jsonl(task_dir / "judge_holdout_transcripts.jsonl", strip_internal(holdout))

    style_counter = [0]
    judge_calib_records = []
    for t in calib:
        judge_calib_records.extend(build_judge_summaries(rng, t, style_counter, "jc"))
    write_jsonl(task_dir / "judge_calib.jsonl", judge_calib_records)

    style_counter = [0]
    judge_holdout_records = []
    for t in holdout:
        judge_holdout_records.extend(build_judge_summaries(rng, t, style_counter, "jh"))
    write_jsonl(task_dir / "judge_holdout.jsonl", judge_holdout_records)

    card = DatasetCard(
        source="synthetic",
        licence="Apache-2.0",
        seed=seed,
        metric="pass_rate",
        distinct_templates=(
            len(MEETING_TOPICS)
            + len(WORKSTREAMS)
            + len(DUE_DATES)
            + len(AMOUNT_VALUES)
            + len(DECISION_ACTIONS)
            + len(TASK_ACTIONS)
            + len(ALT_APPROACHES)
            + len(OPENING_TURNS)
            + len(CLOSING_TURNS)
            + len(SMALL_TALK)
            + len(SUPERSEDED)
            + len(REJECTED)
            + len(ELABORATIONS)
            + sum(len(v) for v in SUMMARY_STYLES.values())
        ),
        notes=(
            "600-1000 word meeting transcripts with 4-6 required facts and 5-10 distractors; "
            "judge-calib/judge-holdout summaries are construction-labelled, not human-labelled."
        ),
    )
    spec = TaskSpec(
        name="summarisation",
        kind="summarisation",
        description="Meeting transcript to an action summary.",
        calib="calib.jsonl",
        test="test.jsonl",
        max_words=120,
        card=card,
    )
    write_yaml_task(task_dir / "task.yaml", spec)
    return card


# --------------------------------------------------------------------------------------
# entity_matching
# --------------------------------------------------------------------------------------

ENTITY_COUNTRIES = ["GB", "US", "CA"]
TRANSLITERATION_PAIRS = [
    ("Müller", "Mueller"),
    ("Björk", "Bjork"),
    ("Søren", "Soren"),
    ("François", "Francois"),
    ("Håkon", "Hakon"),
]
QUALIFIERS = ["International", "Group", "& Partners", "Holdings"]
QUALIFIER_ABBR = {"International": "Intl", "Group": "Grp", "& Partners": "& Ptnrs", "Holdings": "Hldgs"}
LEGAL_FORM_LONG = {"Ltd": "Limited", "LLP": "Limited Liability Partnership", "Inc.": "Incorporated", "LLC": "L.L.C."}
REGIONS = ["European", "North American", "APAC", "UK", "Nordic"]


def public_record(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def make_master(rng: random.Random) -> dict[str, Any]:
    country = rng.choice(ENTITY_COUNTRIES)
    sector = rng.choice(SECTORS)
    surname_style = rng.random() < 0.2
    base, base_ascii = rng.choice(TRANSLITERATION_PAIRS) if surname_style else (rng.choice(COMPANY_PREFIXES),) * 2
    qualifier = rng.choice(QUALIFIERS) if rng.random() < 0.4 else None
    suffix = rng.choice(SECTOR_SUFFIXES[sector])
    legal = rng.choice(LEGAL_FORMS[country])
    bits = [base, qualifier, suffix, legal]
    name = " ".join(b for b in bits if b)
    check_no_blocklisted_brand(name)
    street, city = address_line(rng, country)
    slug = slugify(f"{base_ascii} {suffix}")
    tld = rng.choice(["com", "org", "net"])
    return {
        "name": name,
        "street": street,
        "postcode": postcode_for(rng, country),
        "city": city,
        "country": country,
        "domain": f"{slug}.example.{tld}",
        "vat_id": f"{country}{rng.randint(100000000, 999999999)}",
        "phone": phone_for_country(rng, country)[0],
        "_base": base,
        "_base_ascii": base_ascii,
        "_qualifier": qualifier,
        "_suffix": suffix,
        "_legal": legal,
        "_surname_style": surname_style,
    }


def transform_legal_form(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    long_form = LEGAL_FORM_LONG.get(m["_legal"])
    new_legal = long_form if long_form and rng.random() < 0.7 else m["_legal"]
    bits = [m["_base"], m["_qualifier"], m["_suffix"], new_legal]
    r["name"] = " ".join(b for b in bits if b)
    return r


def transform_abbreviation(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    abbr = QUALIFIER_ABBR[m["_qualifier"]]
    bits = [m["_base"], abbr, m["_suffix"], m["_legal"]]
    r["name"] = " ".join(b for b in bits if b)
    return r


def transform_typo(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    name = m["name"]
    candidates = [i for i in range(len(name) - 1) if name[i] != " " and name[i + 1] != " "]
    pos = rng.choice(candidates) if candidates else 0
    chars = list(name)
    if rng.random() < 0.5:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    else:
        del chars[pos]
    r["name"] = "".join(chars)
    return r


def transform_transliteration(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    r["name"] = m["name"].replace(m["_base"], m["_base_ascii"])
    return r


def transform_missing_fields(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    for field in rng.sample(["phone", "street"], k=rng.choice([1, 2])):
        r[field] = None
    return r


def transform_moved_office(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    street, city = address_line(rng, m["country"])
    r["street"], r["city"], r["postcode"] = street, city, postcode_for(rng, m["country"])
    return r


def negative_other_country(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    country = rng.choice([c for c in ENTITY_COUNTRIES if c != m["country"]])
    street, city = address_line(rng, country)
    return {
        "name": m["name"],
        "street": street,
        "city": city,
        "postcode": postcode_for(rng, country),
        "country": country,
        "domain": m["domain"],
        "vat_id": f"{country}{rng.randint(100000000, 999999999)}",
        "phone": phone_for_country(rng, country)[0],
    }


def negative_parent_subsidiary(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    r = dict(m)
    r["name"] = f"{m['name']} ({rng.choice(REGIONS)} Division)"
    r["vat_id"] = f"{m['country']}{rng.randint(100000000, 999999999)}"
    street, city = address_line(rng, m["country"])
    r["street"], r["city"], r["postcode"] = street, city, postcode_for(rng, m["country"])
    return r


def negative_similar_name_same_street(rng: random.Random, m: dict[str, Any]) -> dict[str, Any]:
    other = rng.choice([p for p in COMPANY_PREFIXES if p != m["_base"]])
    name = " ".join([other, m["_suffix"], m["_legal"]])
    check_no_blocklisted_brand(name)
    slug = slugify(f"{other} {m['_suffix']}")
    tld = rng.choice(["com", "org", "net"])
    return {
        "name": name,
        "street": m["street"],
        "city": m["city"],
        "postcode": m["postcode"],
        "country": m["country"],
        "domain": f"{slug}.example.{tld}",
        "vat_id": f"{m['country']}{rng.randint(100000000, 999999999)}",
        "phone": phone_for_country(rng, m["country"])[0],
    }


POSITIVE_TRANSFORMS = [
    ("legal_form_variant", transform_legal_form),
    ("abbreviation", transform_abbreviation),
    ("typo", transform_typo),
    ("transliteration", transform_transliteration),
    ("missing_fields", transform_missing_fields),
    ("moved_office", transform_moved_office),
]
NEGATIVE_TRANSFORMS = [
    ("other_country", negative_other_country),
    ("parent_subsidiary", negative_parent_subsidiary),
    ("similar_name_same_street", negative_similar_name_same_street),
]


def build_entity_item(rng: random.Random, item_id: str, match: bool) -> dict[str, Any]:
    master = make_master(rng)
    if match:
        options = [
            (k, fn)
            for k, fn in POSITIVE_TRANSFORMS
            if (k != "abbreviation" or master["_qualifier"]) and (k != "transliteration" or master["_surname_style"])
        ]
        _, fn = rng.choice(options)
        right = fn(rng, master)
    else:
        if rng.random() < 0.45:
            right = make_master(rng)
        else:
            _, fn = rng.choice(NEGATIVE_TRANSFORMS)
            right = fn(rng, master)
    return {"id": item_id, "left": public_record(master), "right": public_record(right), "match": match}


def build_entity_split(rng: random.Random, n: int, prefix: str) -> list[dict[str, Any]]:
    n_match = round(n * 0.4)
    flags = [True] * n_match + [False] * (n - n_match)
    rng.shuffle(flags)
    return [build_entity_item(rng, f"{prefix}-{i:04d}", m) for i, m in enumerate(flags)]


def build_entity_matching(seed: int, out_dir: Path) -> DatasetCard:
    rng = random.Random(f"entity_matching:{seed}")
    calib = build_entity_split(rng, 100, "entity-calib")
    test = build_entity_split(rng, 200, "entity-test")

    task_dir = out_dir / "entity_matching"
    write_jsonl(task_dir / "calib.jsonl", calib)
    write_jsonl(task_dir / "test.jsonl", test)

    card = DatasetCard(
        source="synthetic",
        licence="Apache-2.0",
        seed=seed,
        metric="f1",
        distinct_templates=len(POSITIVE_TRANSFORMS) + len(NEGATIVE_TRANSFORMS) + 1,
        notes="Vendor/customer master dedupe pairs, about 40% matches.",
    )
    spec = TaskSpec(
        name="entity_matching",
        kind="entity_matching",
        description="Two company records to a match/no-match decision.",
        calib="calib.jsonl",
        test="test.jsonl",
        card=card,
    )
    write_yaml_task(task_dir / "task.yaml", spec)
    return card


# --------------------------------------------------------------------------------------
# classification (BANKING77)
# --------------------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        dest.write_bytes(resp.read())


def ensure_banking77_files(banking77_dir: Path) -> tuple[Path, Path, Path]:
    banking77_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path, license_path = (
        banking77_dir / "train.csv",
        banking77_dir / "test.csv",
        banking77_dir / "LICENSE",
    )
    if not train_path.exists():
        _download(BANKING77_TRAIN_URL, train_path)
    if not test_path.exists():
        _download(BANKING77_TEST_URL, test_path)
    if not license_path.exists():
        _download(BANKING77_LICENSE_URL, license_path)
    train_sha, test_sha = sha256_of(train_path), sha256_of(test_path)
    if train_sha != BANKING77_TRAIN_SHA256:
        raise ValueError(f"train.csv sha256 mismatch: got {train_sha}, expected {BANKING77_TRAIN_SHA256}")
    if test_sha != BANKING77_TEST_SHA256:
        raise ValueError(f"test.csv sha256 mismatch: got {test_sha}, expected {BANKING77_TEST_SHA256}")
    return train_path, test_path, license_path


def _read_banking_csv(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        return [(row[0], row[1]) for row in reader]


def default_banking77_dir() -> Path:
    return Path(tempfile.gettempdir()) / "local-enough-banking77-cache"


def build_classification(seed: int, out_dir: Path, banking77_dir: Path | None) -> DatasetCard:
    banking77_dir = banking77_dir or default_banking77_dir()
    train_path, test_path, license_path = ensure_banking77_files(banking77_dir)
    train_rows = _read_banking_csv(train_path)
    test_rows = _read_banking_csv(test_path)

    by_label: dict[str, list[str]] = {}
    for text, label in test_rows:
        by_label.setdefault(label, []).append(text)
    labels = sorted(by_label)

    rng = random.Random(f"classification:{seed}")
    calib_records, test_records = [], []
    for label in labels:
        texts = by_label[label][:]
        rng.shuffle(texts)
        calib_records.extend({"text": t, "label": label} for t in texts[:2])
        test_records.extend({"text": t, "label": label} for t in texts[2:6])
    rng.shuffle(calib_records)
    rng.shuffle(test_records)
    for i, r in enumerate(calib_records):
        r["id"] = f"banking77-calib-{i:04d}"
    for i, r in enumerate(test_records):
        r["id"] = f"banking77-test-{i:04d}"
    train_records = [
        {"id": f"banking77-train-{i:04d}", "text": t, "label": lbl} for i, (t, lbl) in enumerate(train_rows)
    ]

    task_dir = out_dir / "classification"
    write_jsonl(task_dir / "calib.jsonl", calib_records)
    write_jsonl(task_dir / "test.jsonl", test_records)
    write_jsonl(task_dir / "train.jsonl", train_records)
    license_dest = task_dir / "LICENSE-BANKING77.txt"
    license_dest.parent.mkdir(parents=True, exist_ok=True)
    license_dest.write_bytes(license_path.read_bytes())

    card = DatasetCard(
        source="BANKING77 (PolyAI); Casanueva, Temcinas, Gerz, Henderson, Vulic (2020)",
        licence="CC-BY-4.0",
        seed=seed,
        metric="accuracy",
        notes=(
            f"calib=2/intent and test=4/intent sampled disjoint from the official test split (154/308 rows); "
            f"train.jsonl is the full 10,003-row train split. Source sha256: train.csv={BANKING77_TRAIN_SHA256}, "
            f"test.csv={BANKING77_TEST_SHA256}."
        ),
    )
    spec = TaskSpec(
        name="classification",
        kind="classification",
        description="Banking customer intent classification (BANKING77).",
        calib="calib.jsonl",
        test="test.jsonl",
        train="train.jsonl",
        labels=labels,
        card=card,
    )
    write_yaml_task(task_dir / "task.yaml", spec)
    return card


# --------------------------------------------------------------------------------------
# bring-your-own example: a tiny 3-label support-ticket classifier
# --------------------------------------------------------------------------------------

CUSTOM_LABELS = ["billing", "bug_report", "how_to"]
CUSTOM_FEATURES = [
    "the dashboard",
    "the export tool",
    "the mobile app",
    "the notification settings",
    "the search filter",
    "the billing page",
    "the API integration",
    "the calendar sync",
    "the file upload",
    "the user roles panel",
]
CUSTOM_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {feature} subscription this month, can you refund the extra payment?",
        "My invoice shows an amount I don't recognise, can someone explain the charge for {feature}?",
        "I'd like to update my payment method before the next billing cycle.",
        "Can you confirm when my next invoice for {feature} is due?",
        "I cancelled my plan last week but was still billed, please look into this.",
        "The discount code I applied didn't reduce the total on my invoice.",
        "I need a copy of last month's invoice for our accounting team.",
        "Is it possible to switch from monthly to annual billing?",
        "My card was declined but the charge still shows as pending, what should I do?",
        "We'd like to add a second seat to our plan, how does the pro-rated charge work?",
    ],
    "bug_report": [
        "{feature} keeps crashing whenever I try to save changes.",
        "I'm getting an error message when I open {feature}, nothing loads.",
        "{feature} shows the wrong data compared to what's in my account.",
        "After the last update, {feature} stopped working entirely.",
        "I can't log in anymore, the page just spins and never loads.",
        "{feature} is extremely slow today, taking over a minute to respond.",
        "There's a broken link on {feature} that leads to a missing page.",
        "My changes in {feature} aren't being saved between sessions.",
        "{feature} throws a permissions error even though I'm an admin.",
        "The app crashed twice today while I was using {feature}.",
    ],
    "how_to": [
        "How do I set up {feature} for my whole team?",
        "Is there a way to export data from {feature} to a spreadsheet?",
        "Can you point me to documentation on configuring {feature}?",
        "What's the best way to migrate our data into {feature}?",
        "How can I give a colleague access to {feature} without making them an admin?",
        "Is it possible to customise the layout of {feature}?",
        "How do I connect {feature} to our existing tools?",
        "Where can I change the default settings for {feature}?",
        "How do I reset {feature} back to its original configuration?",
        "Can you walk me through enabling notifications for {feature}?",
    ],
}


def build_custom_split(rng: random.Random, n_per_label: int, prefix: str) -> list[dict[str, Any]]:
    records = []
    for label in CUSTOM_LABELS:
        for _ in range(n_per_label):
            tmpl = rng.choice(CUSTOM_TEMPLATES[label])
            text = tmpl.format(feature=rng.choice(CUSTOM_FEATURES)) if "{feature}" in tmpl else tmpl
            records.append({"text": text, "label": label})
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"{prefix}-{i:04d}"
    return records


def build_custom_example(seed: int, out_dir: Path) -> DatasetCard:
    rng = random.Random(f"custom_support_tickets:{seed}")
    calib = build_custom_split(rng, 10, "custom-calib")
    test = build_custom_split(rng, 20, "custom-test")
    train = build_custom_split(rng, 50, "custom-train")

    write_jsonl(out_dir / "calib.jsonl", calib)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "train.jsonl", train)

    card = DatasetCard(
        source="synthetic example",
        licence="Apache-2.0",
        seed=seed,
        metric="accuracy",
        distinct_templates=sum(len(v) for v in CUSTOM_TEMPLATES.values()),
        notes="Tiny made-up 3-label support-ticket classifier demonstrating bring-your-own-task.",
    )
    spec = TaskSpec(
        name="custom_support_tickets",
        kind="classification",
        description="Example bring-your-own task: a 3-label support-ticket classifier.",
        calib="calib.jsonl",
        test="test.jsonl",
        train="train.jsonl",
        labels=CUSTOM_LABELS,
        card=card,
    )
    write_yaml_task(out_dir / "task.yaml", spec)
    return card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--banking77-dir", type=Path, default=None)
    parser.add_argument("--examples-out", type=Path, default=DEFAULT_EXAMPLES_OUT)
    parser.add_argument(
        "--skip-banking77", action="store_true", help="Skip the classification task (needs network access once)."
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    build_extraction(args.seed, args.out)
    print("extraction: ok")
    build_pii(args.seed, args.out)
    print("pii_redaction: ok")
    build_summarisation(args.seed, args.out)
    print("summarisation: ok")
    build_entity_matching(args.seed, args.out)
    print("entity_matching: ok")
    if args.skip_banking77:
        print("classification: skipped")
    else:
        build_classification(args.seed, args.out, args.banking77_dir)
        print("classification: ok")
    build_custom_example(args.seed, args.examples_out)
    print("custom-task example: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
