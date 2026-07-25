from __future__ import annotations

import re


MAX_USER_MESSAGE_CHARS = 2000


def apply_input_guardrails(user_message: str) -> str:
    text = (user_message or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > MAX_USER_MESSAGE_CHARS:
        text = text[:MAX_USER_MESSAGE_CHARS].strip()
    return text


def apply_output_guardrails(response_text: str) -> str:
    text = (response_text or "").strip()
    if not text:
        return "I can help. Please share the customer details and I will guide you step by step."

    # Block raw tool-call syntax in end-user visible messages.
    if re.search(r"\b[a-z_]+\([^\n\r)]*\)", text):
        return (
            "I completed the workflow and recorded the required interaction details. "
            "I can also provide a clean summary if you want to share this with the customer."
        )

    looks_like_json = (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
        or '"tool"' in text
    )
    if looks_like_json:
        return "I can help with retention guidance. Please share the customer context so I can provide a clear summary."

    return text


def normalize_customer_id(raw_customer_id: str) -> str:
    source = (raw_customer_id or "").upper().strip()

    tc_match = re.search(r"\bTC[\s_-]?(\d{6})\b", source)
    if tc_match:
        return f"TC-{tc_match.group(1)}"

    c_match = re.search(r"\bC(\d{3,8})\b", source)
    if c_match:
        return f"C{c_match.group(1)}"

    return source


def extract_customer_id(text: str) -> str | None:
    source = (text or "").upper()
    candidates = [
        r"\bTC[\s_-]?\d{6}\b",
        r"\bC\d{3,8}\b",
    ]
    for pattern in candidates:
        match = re.search(pattern, source)
        if match:
            return normalize_customer_id(match.group(0))
    return None


def detect_out_of_scope(user_message: str) -> tuple[bool, str]:
    text = (user_message or "").lower()
    keywords = [
        "reset modem",
        "restart modem",
        "reboot modem",
        "remote modem",
        "technical support",
        "network outage",
        "line fault",
    ]
    if any(keyword in text for keyword in keywords):
        return (
            True,
            (
                "I cannot perform remote modem or technical operations. "
                "I can help assess churn risk and recommend retention actions. "
                "Please transfer technical issues to technical support."
            ),
        )
    return False, ""


def detect_small_talk_or_greeting(user_message: str) -> tuple[bool, str]:
    text = (user_message or "").strip().lower()
    if not text:
        return True, "Hi! How can I help you today?"

    greeting_only_patterns = [
        r"^(hi|hai|hello|hey|good morning|good afternoon|good evening)$",
        r"^(thanks|thank you|ok|okay|cool|great)$",
    ]
    if any(re.match(pattern, text) for pattern in greeting_only_patterns):
        return (
            True,
            (
                "Hi! How can I help you today? "
                "If this is about a customer case, just share what happened and I will guide you from there."
            ),
        )

    return False, ""


def likely_retention_intent(user_message: str) -> bool:
    text = (user_message or "").lower()
    keywords = [
        "customer",
        "cancel",
        "churn",
        "retention",
        "offer",
        "discount",
        "bill",
        "risk",
        "phone",
        "refund",
    ]
    return any(keyword in text for keyword in keywords)


def detect_escalation(user_message: str) -> tuple[bool, str, str]:
    text = (user_message or "").lower()

    escalation_rules: list[tuple[str, str, list[str]]] = [
        ("legal", "Customer mentioned legal action.", ["legal", "lawyer", "attorney", "sue", "lawsuit"]),
        ("compliance", "Customer raised regulatory/compliance concerns.", ["regulator", "compliance", "gdpr", "privacy complaint"]),
        ("billing_dispute", "Complex billing dispute requires supervisor review.", ["billing dispute", "wrong charge", "refund dispute", "unauthorized charge"]),
        ("fraud", "Fraud or identity-theft concern requires supervisor review.", ["fraud", "identity theft", "stolen identity", "account takeover"]),
        ("abuse", "Abusive or threatening interaction requires supervisor intervention.", ["abuse", "threat", "harassment"]),
        ("customer_request", "Customer explicitly requested a supervisor.", ["supervisor", "manager", "escalate"]),
        ("policy_exception", "High business-risk scenario requires supervisor approval.", ["business risk", "high risk decision", "approval needed"]),
    ]

    for category, reason, keywords in escalation_rules:
        if any(keyword in text for keyword in keywords):
            return True, category, reason

    return False, "", ""
