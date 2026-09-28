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
import random
import re
import sys
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
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)
BANKING77_LICENSE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/LICENSE"
BANKING77_TRAIN_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
BANKING77_TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"

# --------------------------------------------------------------------------------------
# Shared pools and helpers
# --------------------------------------------------------------------------------------

# ~100 well-known real brands: generated names are checked against this list so no
# fictional pool entry can accidentally collide with a real company.
BLOCKLIST_BRANDS = [
    "google", "alphabet", "microsoft", "amazon", "apple", "meta", "facebook", "netflix",
    "tesla", "spacex", "ibm", "oracle", "sap", "salesforce", "adobe", "intel", "amd",
    "nvidia", "samsung", "sony", "lg", "huawei", "xiaomi", "dell", "hp", "hewlett-packard",
    "lenovo", "cisco", "vmware", "servicenow", "workday", "zoom", "slack", "atlassian",
    "shopify", "stripe", "paypal", "visa", "mastercard", "americanexpress", "coca-cola",
    "pepsi", "pepsico", "nestle", "unilever", "procter&gamble", "p&g", "walmart", "target",
    "costco", "ikea", "nike", "adidas", "puma", "reebok", "underarmour", "mcdonalds",
    "burgerking", "kfc", "starbucks", "subway", "dominos", "pizzahut", "fedex", "ups",
    "dhl", "maersk", "boeing", "airbus", "lockheedmartin", "generalelectric", "ge",
    "siemens", "bosch", "3m", "caterpillar", "johndeere", "toyota", "honda", "ford",
    "generalmotors", "gm", "volkswagen", "bmw", "mercedes-benz", "audi", "renault",
    "hyundai", "kia", "chevron", "exxonmobil", "exxon", "shell", "bp", "totalenergies",
    "deloitte", "pwc", "ey", "kpmg", "mckinsey", "accenture", "goldmansachs", "jpmorgan",
    "citigroup", "hsbc", "barclays", "santander", "ups", "airbnb", "uber", "lyft",
    "linkedin", "twitter", "x-corp", "tiktok", "bytedance", "snapchat", "pinterest",
    "reddit", "wikipedia", "ebay", "alibaba", "tencent", "baidu", "spotify", "disney",
    "warnerbros", "universal", "sony-pictures", "verizon", "at&t", "tmobile", "vodafone",
    "orange", "bt", "sky", "samsungelectronics", "canon", "nikon", "panasonic",
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
    for forbidden in ("andrii", "boiko"):
        if forbidden in name.lower():
            raise AssertionError(f"generated name {name!r} contains forbidden real name {forbidden!r}")


FIRST_NAMES = [
    "Priya", "Marcus", "Elena", "Tomasz", "Aisha", "Noah", "Freya", "Kwame", "Sofia",
    "Liam", "Ingrid", "Dmitri", "Hana", "Callum", "Yuki", "Rosa", "Declan", "Mei",
    "Owen", "Zara", "Felix", "Amara", "Bjorn", "Nadia", "Theo", "Layla", "Gareth",
    "Ines", "Milo", "Chiara", "Anders", "Fatima", "Jonas", "Wren", "Tariq", "Esme",
    "Rafael", "Ottilie", "Kenji", "Maya",
]
LAST_NAMES = [
    "Hartley", "Novak", "Oduya", "Berglund", "Castellano", "Whitfield", "Moreau",
    "Lindqvist", "Okafor", "Bergman", "Sorensen", "Delacroix", "Marchetti", "Kowalski",
    "Renner", "Abara", "Fenwick", "Larkin", "Vasquez", "Holloway", "Sato", "Brandt",
    "Costa", "Nakamura", "Hendricks", "Solberg", "Iqbal", "Tremblay", "Weiss", "Duarte",
    "Ashworth", "Callahan", "Ferraro", "Mbeki", "Santini", "Boucher", "Ravensworth",
    "Adeyemi", "Lindgren", "Petrova",
]

COMPANY_PREFIXES = [
    "Harrow", "Kestrel", "Brightfield", "Thorncliff", "Meridian", "Oakstead",
    "Silverline", "Cobalt", "Larkspur", "Fenwick", "Ashgrove", "Northgate", "Amberwood",
    "Foxglove", "Hazelmere", "Sterling Vale", "Copperfield", "Elmridge", "Ravenscroft",
    "Wrenfield", "Moorbank", "Sandpiper", "Briarwood", "Cinderford", "Thistledown",
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
    "Elm Street", "Victoria Road", "Kings Parade", "Mill Lane", "Harbour View",
    "Chapel Row", "Foundry Street", "Orchard Close", "Station Approach", "Bridge Street",
    "Maple Avenue", "Riverside Drive", "Commerce Way", "Union Street", "Wellington Road",
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
        yaml.safe_dump(data, fh, sort_keys=True, allow_unicode=False, default_flow_style=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
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
        return f"{area}{rng.randint(1, 20)} {rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
    if country == "US":
        return f"{rng.randint(10000, 99999)}"
    letters = "ABCEGHJKLMNPRSTVXY"
    return f"{rng.choice(letters)}{rng.randint(0, 9)}{rng.choice(letters)} {rng.randint(0, 9)}{rng.choice(letters)}{rng.randint(0, 9)}"


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
    "I'm reaching out on behalf of {company}; we need a replacement for our current tooling and {product} looks promising.",
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
        f"Hi there,",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--banking77-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    build_extraction(args.seed, args.out)
    print("extraction: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
