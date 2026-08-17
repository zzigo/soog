"""Small Logto auth, render metering, quotas, and admin reporting for SOOG."""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import jwt
from flask import g, jsonify, make_response, request
from jwt import PyJWKClient


BASE_DIR = Path(__file__).resolve().parent
USAGE_DB_PATH = Path(
    os.getenv("SOOG_USAGE_DB", str(BASE_DIR / "offload" / "usage.sqlite3"))
).expanduser()
LOGTO_ISSUER = os.getenv("LOGTO_ISSUER_URL", "https://logto.zztt.org/oidc").rstrip("/")
LOGTO_JWKS_URL = os.getenv("LOGTO_JWKS_URL", f"{LOGTO_ISSUER}/jwks").strip()
LOGTO_JWKS_USER_AGENT = os.getenv(
    "LOGTO_JWKS_USER_AGENT", "SOOG-Backend/1.0"
).strip()
LOGTO_APP_ID = os.getenv("SOOG_LOGTO_APP_ID", "").strip()
LOGTO_API_RESOURCE = os.getenv(
    "SOOG_LOGTO_API_RESOURCE", "https://soog.zztt.org/api"
).strip()
LOGTO_SIGNING_ALGORITHMS = [
    value.strip()
    for value in os.getenv("LOGTO_SIGNING_ALGORITHMS", "ES384").split(",")
    if value.strip()
]
DEFAULT_DAILY_QUOTA = max(0, int(os.getenv("SOOG_DEFAULT_DAILY_RENDER_QUOTA", "10") or 10))
DEFAULT_WEEKLY_QUOTA = max(0, int(os.getenv("SOOG_DEFAULT_WEEKLY_RENDER_QUOTA", "40") or 40))
USAGE_TIMEZONE_NAME = os.getenv("SOOG_USAGE_TIMEZONE", "Europe/Zurich").strip() or "Europe/Zurich"
ADMIN_SUBJECTS = {
    value.strip()
    for value in os.getenv("SOOG_ADMIN_SUBJECTS", "").split(",")
    if value.strip()
}

try:
    USAGE_TIMEZONE = ZoneInfo(USAGE_TIMEZONE_NAME)
except Exception:
    USAGE_TIMEZONE = dt.timezone.utc

_jwks_client: Optional[PyJWKClient] = None


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401, code: str = "unauthorized"):
        super().__init__(message)
        self.status = status
        self.code = code


def _connect() -> sqlite3.Connection:
    USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(USAGE_DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_usage_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_users (
                subject TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                daily_quota INTEGER,
                weekly_quota INTEGER
            );

            CREATE TABLE IF NOT EXISTS render_events (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                action TEXT NOT NULL,
                prompt_hash TEXT,
                prompt_chars INTEGER NOT NULL DEFAULT 0,
                request_id TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                http_status INTEGER,
                duration_ms INTEGER,
                charged INTEGER NOT NULL DEFAULT 1,
                llm_calls INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                provider_calls_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                FOREIGN KEY(subject) REFERENCES usage_users(subject)
            );

            CREATE INDEX IF NOT EXISTS idx_render_events_started
                ON render_events(started_at);
            CREATE INDEX IF NOT EXISTS idx_render_events_subject_started
                ON render_events(subject, started_at);
            """
        )


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat()


def _period_bounds(now: Optional[dt.datetime] = None) -> Dict[str, dt.datetime]:
    local_now = (now or _utc_now()).astimezone(USAGE_TIMEZONE)
    day_start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start_local = day_start_local - dt.timedelta(days=day_start_local.weekday())
    return {
        "day_start": day_start_local.astimezone(dt.timezone.utc),
        "day_end": (day_start_local + dt.timedelta(days=1)).astimezone(dt.timezone.utc),
        "week_start": week_start_local.astimezone(dt.timezone.utc),
        "week_end": (week_start_local + dt.timedelta(days=7)).astimezone(dt.timezone.utc),
    }


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(
            LOGTO_JWKS_URL,
            cache_keys=True,
            lifespan=3600,
            headers={"User-Agent": LOGTO_JWKS_USER_AGENT},
        )
    return _jwks_client


def _decode_token(token: str, audience: str) -> Dict[str, Any]:
    if not token:
        raise AuthError("Authentication required")
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=LOGTO_SIGNING_ALGORITHMS,
            audience=audience,
            issuer=LOGTO_ISSUER,
            options={"require": ["exp", "iss", "sub"]},
        )
    except Exception as error:
        # Keep enough claim metadata to diagnose issuer/audience/key drift in
        # production without ever logging the bearer token itself.
        try:
            header = jwt.get_unverified_header(token)
            unverified = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": False,
                    "verify_aud": False,
                    "verify_iss": False,
                },
            )
            logging.warning(
                "SOOG auth token rejected (%s): %s; alg=%r kid=%r iss=%r aud=%r sub=%r exp=%r expected_aud=%r",
                type(error).__name__,
                error,
                header.get("alg"),
                header.get("kid"),
                unverified.get("iss"),
                unverified.get("aud"),
                unverified.get("sub"),
                unverified.get("exp"),
                audience,
            )
        except Exception as metadata_error:
            logging.warning(
                "SOOG auth token rejected (%s): %s; token metadata unavailable (%s)",
                type(error).__name__,
                error,
                metadata_error,
            )
        raise AuthError(f"Invalid or expired access token: {error}") from error
    return dict(claims)


def authenticate_request() -> Dict[str, Any]:
    authorization = request.headers.get("Authorization", "")
    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise AuthError("Sign in to render with SOOG")

    id_token = request.headers.get("X-SOOG-ID-Token", "").strip()
    profile_claims: Dict[str, Any] = {}
    try:
        claims = _decode_token(access_token.strip(), LOGTO_API_RESOURCE)
    except AuthError:
        # A valid SOOG ID token is still strong proof of the same interactive
        # login. Accept it as a narrow fallback so a Logto resource-token
        # format mismatch cannot hide the admin UI or block render metering.
        if not id_token or not LOGTO_APP_ID:
            raise
        profile_claims = _decode_token(id_token, LOGTO_APP_ID)
        claims = profile_claims
        logging.warning("SOOG auth accepted a validated ID-token fallback")

    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise AuthError("Access token does not identify a user")

    if not profile_claims and id_token and LOGTO_APP_ID:
        try:
            candidate = _decode_token(id_token, LOGTO_APP_ID)
            if str(candidate.get("sub") or "") == subject:
                profile_claims = candidate
        except AuthError:
            profile_claims = {}

    identity = {
        "subject": subject,
        "email": str(profile_claims.get("email") or claims.get("email") or "").strip(),
        "name": str(
            profile_claims.get("name")
            or profile_claims.get("username")
            or claims.get("name")
            or ""
        ).strip(),
        "is_admin": subject in ADMIN_SUBJECTS,
    }
    g.soog_identity = identity
    return identity


def _touch_user(connection: sqlite3.Connection, identity: Dict[str, Any]) -> None:
    now = _iso(_utc_now())
    connection.execute(
        """
        INSERT INTO usage_users(subject, email, name, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(subject) DO UPDATE SET
            email = CASE WHEN excluded.email <> '' THEN excluded.email ELSE usage_users.email END,
            name = CASE WHEN excluded.name <> '' THEN excluded.name ELSE usage_users.name END,
            last_seen_at = excluded.last_seen_at
        """,
        (identity["subject"], identity.get("email", ""), identity.get("name", ""), now, now),
    )


def _effective_quota(row: Optional[sqlite3.Row], field: str, default: int, is_admin: bool) -> Optional[int]:
    if is_admin:
        return None
    raw = row[field] if row is not None else None
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        return None
    return value


def _quota_snapshot(connection: sqlite3.Connection, identity: Dict[str, Any]) -> Dict[str, Any]:
    bounds = _period_bounds()
    row = connection.execute(
        "SELECT daily_quota, weekly_quota FROM usage_users WHERE subject = ?",
        (identity["subject"],),
    ).fetchone()
    daily_used = int(
        connection.execute(
            "SELECT COUNT(*) FROM render_events WHERE subject = ? AND charged = 1 AND started_at >= ?",
            (identity["subject"], _iso(bounds["day_start"])),
        ).fetchone()[0]
    )
    weekly_used = int(
        connection.execute(
            "SELECT COUNT(*) FROM render_events WHERE subject = ? AND charged = 1 AND started_at >= ?",
            (identity["subject"], _iso(bounds["week_start"])),
        ).fetchone()[0]
    )
    daily_limit = _effective_quota(row, "daily_quota", DEFAULT_DAILY_QUOTA, identity["is_admin"])
    weekly_limit = _effective_quota(row, "weekly_quota", DEFAULT_WEEKLY_QUOTA, identity["is_admin"])
    return {
        "daily": {
            "used": daily_used,
            "limit": daily_limit,
            "remaining": None if daily_limit is None else max(0, daily_limit - daily_used),
            "resets_at": _iso(bounds["day_end"]),
        },
        "weekly": {
            "used": weekly_used,
            "limit": weekly_limit,
            "remaining": None if weekly_limit is None else max(0, weekly_limit - weekly_used),
            "resets_at": _iso(bounds["week_end"]),
        },
    }


def get_identity_payload(identity: Dict[str, Any]) -> Dict[str, Any]:
    with _connect() as connection:
        _touch_user(connection, identity)
        quota = _quota_snapshot(connection, identity)
    return {"user": identity, "quota": quota, "timezone": USAGE_TIMEZONE_NAME}


def _quota_exceeded(quota: Dict[str, Any]) -> Optional[str]:
    for period in ("daily", "weekly"):
        item = quota[period]
        if item["limit"] is not None and item["used"] >= item["limit"]:
            return period
    return None


def _request_prompt() -> str:
    body = request.get_json(silent=True) or {}
    return str(body.get("prompt") or body.get("text") or "").strip()


def _start_event(
    identity: Dict[str, Any], action: str, charged: bool
) -> tuple[str, Dict[str, Any]]:
    prompt = _request_prompt()
    if charged and not prompt:
        raise AuthError("No prompt provided", status=400, code="invalid_request")
    event_id = str(uuid.uuid4())
    started_at = _iso(_utc_now())
    request_id = str((request.get_json(silent=True) or {}).get("request_id") or "").strip()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else ""

    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        _touch_user(connection, identity)
        quota = _quota_snapshot(connection, identity)
        exceeded = _quota_exceeded(quota) if charged else None
        if exceeded:
            connection.rollback()
            raise AuthError(
                f"{exceeded.capitalize()} render quota reached",
                status=429,
                code="quota_exceeded",
            )
        connection.execute(
            """
            INSERT INTO render_events(
                id, subject, action, prompt_hash, prompt_chars, request_id,
                started_at, status, charged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                event_id,
                identity["subject"],
                action,
                prompt_hash,
                len(prompt),
                request_id,
                started_at,
                1 if charged else 0,
            ),
        )
        connection.commit()
        if charged:
            quota["daily"]["used"] += 1
            quota["weekly"]["used"] += 1
            for period in ("daily", "weekly"):
                limit = quota[period]["limit"]
                quota[period]["remaining"] = None if limit is None else max(0, limit - quota[period]["used"])
        return event_id, quota
    finally:
        connection.close()


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def record_provider_call(meta: Dict[str, Any]) -> None:
    calls = getattr(g, "soog_provider_calls", None)
    if calls is None:
        return
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    prompt_tokens = _safe_int(
        usage.get("prompt_tokens")
        or usage.get("promptEvalCount")
        or usage.get("prompt_eval_count")
        or meta.get("prompt_eval_count")
    )
    completion_tokens = _safe_int(
        usage.get("completion_tokens")
        or usage.get("evalCount")
        or usage.get("eval_count")
        or meta.get("eval_count")
    )
    total_tokens = _safe_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    calls.append(
        {
            "provider": str(meta.get("provider") or "ollama"),
            "model": str(meta.get("model") or ""),
            "endpoint": str(meta.get("endpoint") or ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )


def _finish_event(event_id: str, status: str, http_status: int, started: float, error: str = "") -> None:
    calls = list(getattr(g, "soog_provider_calls", []) or [])
    prompt_tokens = sum(_safe_int(call.get("prompt_tokens")) for call in calls)
    completion_tokens = sum(_safe_int(call.get("completion_tokens")) for call in calls)
    total_tokens = sum(_safe_int(call.get("total_tokens")) for call in calls)
    with _connect() as connection:
        connection.execute(
            """
            UPDATE render_events SET
                finished_at = ?, status = ?, http_status = ?, duration_ms = ?,
                llm_calls = ?, prompt_tokens = ?, completion_tokens = ?, total_tokens = ?,
                provider_calls_json = ?, error = ?
            WHERE id = ?
            """,
            (
                _iso(_utc_now()),
                status,
                http_status,
                max(0, int((time.monotonic() - started) * 1000)),
                len(calls),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                json.dumps(calls, ensure_ascii=False),
                str(error or "")[:600],
                event_id,
            ),
        )


def _error_response(error: AuthError):
    payload: Dict[str, Any] = {"error": str(error), "code": error.code}
    if error.status == 429:
        try:
            identity = getattr(g, "soog_identity", None) or authenticate_request()
            with _connect() as connection:
                payload["quota"] = _quota_snapshot(connection, identity)
        except Exception:
            pass
    return jsonify(payload), error.status


def require_auth(admin: bool = False):
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            try:
                identity = authenticate_request()
                if admin and not identity["is_admin"]:
                    raise AuthError("Administrator access required", status=403, code="forbidden")
                return view(*args, **kwargs)
            except AuthError as error:
                return _error_response(error)

        return wrapped

    return decorator


def metered_action(action: str, charged: bool = True):
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            try:
                identity = authenticate_request()
                event_id, quota = _start_event(identity, action, charged)
            except AuthError as error:
                return _error_response(error)

            started = time.monotonic()
            g.soog_provider_calls = []
            try:
                response = make_response(view(*args, **kwargs))
                http_status = int(response.status_code)
                event_status = "succeeded" if http_status < 400 else "failed"
                error_text = ""
                if http_status >= 400:
                    payload = response.get_json(silent=True) or {}
                    error_text = str(payload.get("error") or "")
                _finish_event(event_id, event_status, http_status, started, error_text)
                response.headers["X-SOOG-Daily-Remaining"] = str(quota["daily"]["remaining"])
                response.headers["X-SOOG-Weekly-Remaining"] = str(quota["weekly"]["remaining"])
                return response
            except Exception as error:
                _finish_event(event_id, "failed", 500, started, str(error))
                raise

        return wrapped

    return decorator


def update_user_quota(subject: str, daily: Any, weekly: Any) -> Dict[str, Any]:
    def normalize(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        parsed = int(value)
        return -1 if parsed < 0 else parsed

    daily_value = normalize(daily)
    weekly_value = normalize(weekly)
    now = _iso(_utc_now())
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO usage_users(subject, email, name, first_seen_at, last_seen_at, daily_quota, weekly_quota)
            VALUES (?, '', '', ?, ?, ?, ?)
            ON CONFLICT(subject) DO UPDATE SET
                daily_quota = excluded.daily_quota,
                weekly_quota = excluded.weekly_quota
            """,
            (subject, now, now, daily_value, weekly_value),
        )
        row = connection.execute("SELECT * FROM usage_users WHERE subject = ?", (subject,)).fetchone()
    return dict(row) if row else {}


def _event_rows(start: dt.datetime, end: dt.datetime) -> list[sqlite3.Row]:
    with _connect() as connection:
        return connection.execute(
            """
            SELECT e.*, u.email, u.name
            FROM render_events e
            JOIN usage_users u ON u.subject = e.subject
            WHERE e.started_at >= ? AND e.started_at < ?
            ORDER BY e.started_at DESC
            """,
            (_iso(start), _iso(end)),
        ).fetchall()


def _sum_metrics(rows: Iterable[sqlite3.Row]) -> Dict[str, int]:
    items = list(rows)
    return {
        "renders": sum(int(row["charged"] or 0) for row in items),
        "actions": len(items),
        "llm_calls": sum(int(row["llm_calls"] or 0) for row in items),
        "prompt_tokens": sum(int(row["prompt_tokens"] or 0) for row in items),
        "completion_tokens": sum(int(row["completion_tokens"] or 0) for row in items),
        "total_tokens": sum(int(row["total_tokens"] or 0) for row in items),
        "failed": sum(1 for row in items if row["status"] != "succeeded"),
    }


def usage_dashboard(period: str = "week") -> Dict[str, Any]:
    bounds = _period_bounds()
    if period == "day":
        start, end, bucket_count = bounds["day_start"], bounds["day_end"], 24
        bucket_delta = dt.timedelta(hours=1)
        bucket_format = "%H:00"
    else:
        period = "week"
        start, end, bucket_count = bounds["week_start"], bounds["week_end"], 7
        bucket_delta = dt.timedelta(days=1)
        bucket_format = "%a %d"

    rows = _event_rows(start, end)
    by_user: Dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_user.setdefault(str(row["subject"]), []).append(row)

    with _connect() as connection:
        user_rows = connection.execute("SELECT * FROM usage_users ORDER BY last_seen_at DESC").fetchall()
        users = []
        for user in user_rows:
            identity = {
                "subject": user["subject"],
                "email": user["email"] or "",
                "name": user["name"] or "",
                "is_admin": user["subject"] in ADMIN_SUBJECTS,
            }
            metrics = _sum_metrics(by_user.get(str(user["subject"]), []))
            users.append(
                {
                    **identity,
                    **metrics,
                    "configured_quota": {
                        "daily": user["daily_quota"],
                        "weekly": user["weekly_quota"],
                    },
                    "quota": _quota_snapshot(connection, identity),
                    "last_seen_at": user["last_seen_at"],
                }
            )

    local_start = start.astimezone(USAGE_TIMEZONE)
    series = []
    for index in range(bucket_count):
        bucket_start_local = local_start + bucket_delta * index
        bucket_end_local = bucket_start_local + bucket_delta
        bucket_rows = [
            row
            for row in rows
            if bucket_start_local
            <= dt.datetime.fromisoformat(row["started_at"]).astimezone(USAGE_TIMEZONE)
            < bucket_end_local
        ]
        series.append(
            {
                "label": bucket_start_local.strftime(bucket_format),
                **_sum_metrics(bucket_rows),
            }
        )

    summary = _sum_metrics(rows)
    summary["active_users"] = len(by_user)
    recent = [
        {
            "id": row["id"],
            "subject": row["subject"],
            "email": row["email"] or "",
            "name": row["name"] or "",
            "action": row["action"],
            "started_at": row["started_at"],
            "status": row["status"],
            "llm_calls": row["llm_calls"],
            "total_tokens": row["total_tokens"],
            "duration_ms": row["duration_ms"],
        }
        for row in rows[:20]
    ]
    return {
        "period": period,
        "timezone": USAGE_TIMEZONE_NAME,
        "window": {"start": _iso(start), "end": _iso(end)},
        "summary": summary,
        "series": series,
        "users": users,
        "recent": recent,
        "defaults": {"daily": DEFAULT_DAILY_QUOTA, "weekly": DEFAULT_WEEKLY_QUOTA},
    }
