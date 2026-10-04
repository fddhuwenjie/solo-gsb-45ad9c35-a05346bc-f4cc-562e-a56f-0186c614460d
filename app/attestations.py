"""Immutable admission attestations for frozen release batches.

An attestation is the *admission proof* a caller registers once, at a
chosen moment, for a frozen release batch: it pins the caller-supplied
unique ``attestation_id``, the ``batch_id`` and the *expected*
``batch_hash``, plus the exact snapshot inputs (current time, release
time, timezone, workdays/holidays, per-obligation deadline rules, and the
timestamped materials/evidence rules) that prove every component
obligation is fulfilled.

The store reads the frozen batch manifest and recomputes the fulfillment
snapshot **inside one SQLite write transaction**, so the batch hash and
every judged instance come from a single locked read; the attestation is
written only when the expected batch hash equals the manifest hash and
*all* instances are ``satisfied`` (no ``pending``/``overdue``).  Any
invalid snapshot input (422), hash mismatch (422), unknown batch (404)
or unmet obligation (409) rejects the whole request and rolls back -- no
attestation row is ever written.

The same ``attestation_id`` is idempotent only while the normalized
content is identical (an equivalent instant written with another UTC
offset is still identical, because the snapshot normalizes moments to
UTC and every collection is sorted); a reused id with different content
conflicts (409).  Registered attestations are immutable: subsequent
ruleset publications or later materials never rewrite one.
"""

import re

from .common import ClientError, ConflictError, stable_hash

_ATTESTATION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# Snapshot fields copied verbatim from the request and passed to the
# snapshot builder; the snapshot builder normalizes every one of them
# (moments to UTC, every collection sorted), so they fully determine the
# snapshot_hash and the idempotency comparison.
_SNAPSHOT_FIELDS = (
    "now",
    "release_at",
    "timezone",
    "workdays",
    "holidays",
    "obligations",
    "materials",
    "rules",
)


def validate_attestation_body(body):
    """Validate the attestation request envelope (400 on shape errors).

    Returns ``(attestation_id, batch_id, expected_hash, snapshot_body)``.
    Only request-shape rules live here (they hold regardless of database
    state); every content rule -- timestamp/timezone/rule/material
    validity, batch existence, hash correspondence, obligation statuses
    -- is enforced by the snapshot builder inside the store transaction.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")

    attestation_id = body.get("attestation_id")
    if not isinstance(attestation_id, str) or not attestation_id.strip():
        raise ClientError("attestation_id must be a non-empty string")
    attestation_id = attestation_id.strip()
    if not _ATTESTATION_ID_RE.match(attestation_id):
        raise ClientError(
            "attestation_id may contain only letters, digits, dot,"
            " underscore and hyphen (1..128 characters)"
        )

    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ClientError("batch_id must be a non-empty string")
    batch_id = batch_id.strip()
    if not _BATCH_ID_RE.match(batch_id):
        raise ClientError(
            "batch_id may contain only letters, digits, dot, underscore"
            " and hyphen (1..128 characters)"
        )

    expected_hash = body.get("batch_hash")
    if not isinstance(expected_hash, str) or not _HASH_RE.match(expected_hash):
        raise ClientError("batch_hash must be 64 lowercase hex characters")

    snapshot_body = {"batch_id": batch_id}
    for field in _SNAPSHOT_FIELDS:
        if field in body:
            snapshot_body[field] = body[field]
    return attestation_id, batch_id, expected_hash, snapshot_body


def build_attestation(attestation_id, manifest, content, snapshot):
    """Assemble the immutable attestation record from a judged snapshot.

    Called by the store inside the registering write transaction, after
    the expected batch hash and the all-satisfied gate have both passed.
    ``attestation_hash`` covers the batch hash, the adopted rule
    revisions, the judgment summary and the full canonical snapshot
    content (including ``snapshot_hash``) -- everything except the hash
    itself.
    """
    rule_revisions = manifest.get("rule_revisions", [])
    adopted = sorted(
        {
            (r["ruleset"], r["revision"])
            for r in rule_revisions
            if isinstance(r, dict) and r.get("ruleset") and r.get("revision")
        }
    )
    record_content = {
        "attestation_id": attestation_id,
        "batch_id": manifest["batch_id"],
        "batch_hash": manifest["batch_hash"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "adopted_rule_revisions": [
            {"ruleset": ruleset, "revision": revision}
            for ruleset, revision in adopted
        ],
        "summary": dict(snapshot["summary"]),
        "snapshot": content,
    }
    record = dict(record_content)
    record["attestation_hash"] = stable_hash(record_content)
    return record


def require_all_satisfied(snapshot):
    """Gate the write: every component obligation instance must be satisfied.

    A single ``pending`` or ``overdue`` instance rejects the admission as
    a 409 conflict with the offending instance list (sorted) and the
    recomputed summary -- the caller may still read the snapshot from the
    read-only snapshot endpoint, but no attestation may be registered.
    """
    summary = snapshot.get("summary", {})
    unmet_count = summary.get("pending", 0) + summary.get("overdue", 0)
    if unmet_count == 0 and all(
        item.get("status") == "satisfied"
        for item in snapshot.get("instances", [])
    ):
        return
    unmet = sorted(
        (
            {
                "component": item.get("component"),
                "obligation": item.get("obligation"),
                "status": item.get("status"),
                "deadline": item.get("deadline"),
            }
            for item in snapshot.get("instances", [])
            if item.get("status") in ("pending", "overdue")
        ),
        key=lambda u: (u["component"], u["obligation"]),
    )
    raise ConflictError(
        "attestation refused: %d of %d component obligations are not "
        "satisfied (%d pending, %d overdue)"
        % (
            unmet_count,
            summary.get("total", 0),
            summary.get("pending", 0),
            summary.get("overdue", 0),
        ),
        {
            "reason": "obligations-not-satisfied",
            "summary": dict(summary),
            "unmet_instances": unmet,
        },
    )
