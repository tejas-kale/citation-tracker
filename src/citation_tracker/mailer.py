"""Email delivery via the Resend API."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def send_report(
    report_markdown: str,
    from_address: str,
    to_address: str,
    api_key: str,
    subject: str = "Citation Tracker Report",
) -> None:
    """
    Send *report_markdown* as an email via Resend.

    Raises RuntimeError if the API call fails.
    """
    try:
        import resend  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("resend package not installed") from exc

    resend.api_key = api_key

    params: resend.Emails.SendParams = {
        "from": from_address,
        "to": [to_address],
        "subject": subject,
        "text": report_markdown,
    }

    email = resend.Emails.send(params)
    logger.info("Report emailed, id=%s", email.get("id"))
