#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Single-shape tool with three calling modes (inferred from args, no explicit
mode parameter):

  1. DISCOVERY — pass ``query``. Runs FTS5, dedupes hits by session lineage,
     returns top N sessions each with: snippet, ±5 message window around the
     match, plus bookend_start (first 3 user+assistant msgs of session) and
     bookend_end (last 3). Zero LLM cost.

  2. SCROLL — pass ``session_id`` + ``around_message_id``. Returns a window
     of ±window messages centered on the anchor, no FTS5, no bookends. To
     scroll forward / backward, re-anchor on the last / first message id of
     the returned window.

  3. BROWSE — no args. Returns recent sessions chronologically (titles,
     previews, timestamps).

All three modes operate on the SQLite session DB via the FTS5 index and
the get_anchored_view / get_messages_around primitives in hermes_state.
No LLM calls anywhere — every shape returns actual messages from the DB.

History: PR #20238 (JabberELF) seeded a fast/summary dual-mode split; the
toolkit expansion in PR #26419 (yoniebans) added the anchored drill-down,
bookends, and sort. This module merges all of that into a single calling
shape with no mode parameter, no summary LLM path, and explicit scroll
support.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

# Sources that are excluded from session browsing/searching by default.
# Third-party integrations tag their sessions with HERMES_SESSION_SOURCE=tool;
# delegate subagent runs are tagged "subagent" — neither belongs in the
# user's session history.
_HIDDEN_SESSION_SOURCES = ("subagent", "tool")

# Automation sources that are kept searchable but DEMOTED below interactive
# sessions in discover ranking. Cron jobs run on a schedule and accumulate
# large volumes of repetitive vocabulary (recurring project names, dates,
# "session", summaries); under bare BM25 they dominate the top-N FTS rows and
# starve out the user's own interactive sessions, producing "recall blindness"
# where only cron sessions surface (#19434). Demoting — not excluding — keeps
# cron content reachable when it's the only match, while interactive sessions
# always win when both match.
_DEMOTED_SESSION_SOURCES = ("cron",)

# How many FTS rows discover scans before dedup-by-lineage. The interactive
# vs automation split below only helps if enough rows are in hand to find
# interactive matches buried under a wall of cron hits, so this is well above
# the handful of distinct sessions a typical query returns.
_DISCOVER_SCAN_LIMIT = 300


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _resolve_to_parent(db, session_id: str) -> str:
    """Walk parent_session_id chain to the lineage root. Falls back to input on errors."""
    if not session_id:
        return session_id
    visited = set()
    cur = session_id
    while cur and cur not in visited:
        visited.add(cur)
        try:
            s = db.get_session(cur)
            if not s:
                break
            parent = s.get("parent_session_id")
            if not parent:
                break
            cur = parent
        except Exception as e:
            logging.debug("Error resolving parent for %s: %s", cur, e, exc_info=True)
            break
    return cur


def _order_for_recall(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对 FTS 结果行进行稳定排序（Stable-sort），使交互式会话的排名优先于自动化任务。

    在每个分类内部（交互式 vs 降级项），原始的 BM25 ``rank`` 顺序均得以保留 ——
    Python 的排序算法是稳定的，且结果行在传入时已按相关性排好序。
    此操作仅改变跨分类间的顺序：在谱系去重（Lineage dedup）过程中，
    Cron 的命中项绝不会挤占交互式的命中项，
    因此即使在纯 BM25 算法下 Cron 行的排名更高，用户自己的对话也会优先呈现（#19434）。
    当降级行是唯一的匹配项时，它们仍会显示出来。
    """
    return sorted(
        raw_results,
        key=lambda r: 1 if (r.get("source") or "") in _DEMOTED_SESSION_SOURCES else 0,
    )


def _shape_message(m: Dict[str, Any], anchor_id: Optional[int] = None) -> Dict[str, Any]:
    """Slim a message row for the tool response. Keeps content even if empty."""
    entry = {
        "id": m.get("id"),
        "role": m.get("role"),
        "content": m.get("content"),
        "timestamp": m.get("timestamp"),
    }
    if m.get("tool_name"):
        entry["tool_name"] = m.get("tool_name")
    if m.get("tool_calls"):
        entry["tool_calls"] = m.get("tool_calls")
    if m.get("tool_call_id"):
        entry["tool_call_id"] = m.get("tool_call_id")
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    # Strip None values to keep payload tight, but always keep content
    # (absent content is meaningful — tool-call-only assistant turns).
    return {k: v for k, v in entry.items() if v is not None or k in ("content",)}


def _resolve_profile_db(profile: str):
    """Open another profile's ``state.db`` read-only, or None for the current one.

    The desktop's ``@session:<profile>/<id>`` links always carry the source
    profile, so a linked session from profile B can be read while the agent
    runs in profile A. ``read_only=True`` (mode=ro) takes no write lock — safe
    to point at a live profile's DB, including our own. Returns None when no
    profile is given (use the caller's default db).
    """
    if profile is None or not str(profile).strip():
        return None

    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB

    canon = profiles_mod.normalize_profile_name(profile)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise ValueError(f"profile '{canon}' does not exist")

    return SessionDB(db_path=profiles_mod.get_profile_dir(canon) / "state.db", read_only=True)


def _locate_session_db(session_id: str):
    """Scan every profile's ``state.db`` (read-only) for a session id.

    Returns ``(db, profile_name)`` for the first profile that owns the id, or
    ``(None, None)``. Session ids are globally unique (timestamp + random hex),
    so the first hit is authoritative. This is the safety net for linked-session
    reads where the model dropped the owning profile from the link and passed a
    bare id — we find it wherever it actually lives instead of failing.
    """
    from pathlib import Path

    try:
        from hermes_cli import profiles as profiles_mod
        from hermes_state import SessionDB
    except Exception:
        return None, None

    targets = [("default", profiles_mod.get_profile_dir("default"))]
    try:
        targets += [(info.name, info.path) for info in profiles_mod.list_profiles()]
    except Exception:
        logging.debug("list_profiles failed during session locate", exc_info=True)

    seen: set = set()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        key = str(db_path)
        if key in seen or not db_path.exists():
            continue
        seen.add(key)
        try:
            pdb = SessionDB(db_path=db_path, read_only=True)
        except Exception:
            continue
        try:
            if pdb.get_session(session_id):
                return pdb, name
        except Exception:
            logging.debug("get_session probe failed for %s in %s", session_id, name, exc_info=True)
        pdb.close()

    return None, None


def _read_session(db, session_id: str, head: int = 20, tail: int = 10) -> str:
    """读取模式（Read shape）：根据 ID 导出整个会话（内容过大时返回头部与尾部消息）。

    适用于关联会话的情形 —— 当用户发送了一个 @session 引用，
    且 Agent 需要获取对应的对话转录时使用。
    负载受限：较小的会话将完整返回，
    较大的会话则仅返回前 ``head`` 条与后 ``tail`` 条消息，
    并附带滚动浏览中间内容的提示信息。
    """
    try:
        meta = db.get_session(session_id) or {}
    except Exception as e:
        logging.debug("get_session failed for %s: %s", session_id, e, exc_info=True)
        meta = {}
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    try:
        rows = db.get_messages(session_id)
    except Exception as e:
        logging.error("get_messages failed for %s: %s", session_id, e, exc_info=True)
        return tool_error(f"failed to load session: {e}", success=False)

    shaped = [_shape_message(m) for m in rows]
    total = len(shaped)
    truncated = total > head + tail
    window = shaped[:head] + shaped[-tail:] if truncated else shaped

    response = {
        "success": True,
        "mode": "read",
        "session_id": session_id,
        "session_meta": {
            "when": _format_timestamp(meta.get("started_at")),
            "source": meta.get("source"),
            "model": meta.get("model"),
            "title": meta.get("title"),
        },
        "message_count": total,
        "truncated": truncated,
        "messages": window,
    }
    if truncated:
        response["message"] = (
            f"Session has {total} messages; showing first {head} + last {tail}. "
            "Pass around_message_id (any id above) to scroll the middle."
        )
    return json.dumps(response, ensure_ascii=False)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls, no FTS5)."""
    try:
        sessions = db.list_sessions_rich(
            limit=limit + 5,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            order_by_last_active=True,
        )  # fetch extra so we can skip current

        current_root = _resolve_to_parent(db, current_session_id) if current_session_id else None

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child / delegation sessions
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "browse",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Pass a query= to search, or session_id+around_message_id to scroll.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def _scroll(
    db,
    session_id: str,
    around_message_id: int,
    window: int = 5,
    current_session_id: str = None,
) -> str:
    """滚动模式（Scroll shape）：返回以锚点消息为中心的指定窗口消息。

    不使用 FTS5 全文检索，无首尾片段（bookends）—— 仅返回消息切片。
    保留了探索模式（discovery shape）的会话谱系修正机制：
    如果锚点消息不在指定的会话中，但存在于同谱系的子会话中，
    则静默重新绑定（rebind）。
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return tool_error("scroll requires session_id", success=False)
    session_id = session_id.strip()

    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)

    # Window clamp [1, 20]
    if not isinstance(window, int):
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 5
    window = max(1, min(window, 20))

    # 拒绝在当前活动的会话谱系内进行滚动 ——
    # 这些消息已经包含在上下文（Context）中了。
    if current_session_id:
        a_root = _resolve_to_parent(db, session_id)
        c_root = _resolve_to_parent(db, current_session_id)
        if a_root and c_root and a_root == c_root:
            return tool_error(
                "scroll rejected: anchor lives in the current session lineage (already in your active context)",
                success=False,
            )

    # Session existence check
    try:
        session_meta = db.get_session(session_id) or {}
    except Exception as e:
        logging.debug("get_session failed for %s: %s", session_id, e, exc_info=True)
        session_meta = {}
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    # Fetch the window
    try:
        view = db.get_messages_around(session_id, around_message_id, window=window)
    except Exception as e:
        logging.error("get_messages_around failed: %s", e, exc_info=True)
        return tool_error(f"failed to load messages: {e}", success=False)

    messages = view.get("window") or []

    # 谱系重新绑定（Lineage rebind）：
    # 调用方可能将父 session_id 与存在于后代会话中的消息 ID 进行了配对
    # （压缩/委托操作会创建子会话）。
    # 此时需定位到真实所属的会话并重新获取数据。
    rebind_warning = None
    if not messages:
        owning = None
        try:
            conn = getattr(db, "_conn", None)
            if conn is not None:
                row = conn.execute(
                    "SELECT session_id FROM messages WHERE id = ?",
                    (around_message_id,),
                ).fetchone()
                owning = row[0] if row else None
        except Exception as e:
            logging.debug("owning-session lookup failed: %s", e, exc_info=True)
            owning = None
        if owning and owning != session_id:
            a_root = _resolve_to_parent(db, session_id)
            o_root = _resolve_to_parent(db, owning)
            if a_root and o_root and a_root == o_root:
                try:
                    rebind_view = db.get_messages_around(owning, around_message_id, window=window)
                    messages = rebind_view.get("window") or []
                    if messages:
                        view = rebind_view
                        rebind_warning = (
                            f"around_message_id {around_message_id} lives in {owning} "
                            f"(child of {session_id}); rebound transparently"
                        )
                        try:
                            session_meta = db.get_session(owning) or session_meta
                        except Exception:
                            pass
                        session_id = owning
                except Exception as e:
                    logging.debug("rebind get_messages_around failed: %s", e, exc_info=True)

    if not messages:
        return tool_error(
            f"around_message_id {around_message_id} not in session_id {session_id}",
            success=False,
        )

    response = {
        "success": True,
        "mode": "scroll",
        "session_id": session_id,
        "around_message_id": around_message_id,
        "session_meta": {
            "when": _format_timestamp(session_meta.get("started_at")),
            "source": session_meta.get("source"),
            "model": session_meta.get("model"),
            "title": session_meta.get("title"),
        },
        "window": window,
        "messages": [_shape_message(m, anchor_id=around_message_id) for m in messages],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", 0),
    }
    if rebind_warning:
        response["warning"] = rebind_warning
    return json.dumps(response, ensure_ascii=False)


def _normalize_title_query(query: str) -> str:
    """Strip common quoting the model may include around a remembered title."""
    return query.strip().strip("`'\"")


def _title_match_result(
    db,
    query: str,
    current_lineage_root: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return a discovery-shaped result when the query matches a session title."""
    title_query = _normalize_title_query(query)
    if not title_query:
        return None

    try:
        session_id = db.resolve_session_by_title(title_query)
    except Exception:
        logging.debug("resolve_session_by_title failed for %r", title_query, exc_info=True)
        return None
    if not session_id:
        return None

    lineage_root = _resolve_to_parent(db, session_id)
    if current_lineage_root and lineage_root == current_lineage_root:
        return None

    try:
        session_meta = db.get_session(lineage_root) or db.get_session(session_id) or {}
    except Exception:
        logging.debug("get_session failed for title match %s", session_id, exc_info=True)
        session_meta = {}
    if session_meta.get("source") in _HIDDEN_SESSION_SOURCES:
        return None

    try:
        messages = db.get_messages(session_id)
    except Exception:
        logging.debug("get_messages failed for title match %s", session_id, exc_info=True)
        messages = []

    anchor_id = messages[0].get("id") if messages else None
    if anchor_id is not None:
        try:
            view = db.get_anchored_view(session_id, anchor_id, window=5, bookend=3)
        except Exception:
            logging.debug("get_anchored_view failed for title match %s/%s", session_id, anchor_id, exc_info=True)
            view = {}
    else:
        view = {}

    entry = {
        "session_id": session_id,
        "when": _format_timestamp(session_meta.get("started_at")),
        "source": session_meta.get("source", "unknown"),
        "model": session_meta.get("model") or "unknown",
        "title": session_meta.get("title") or title_query,
        "matched_role": "session_title",
        "match_message_id": anchor_id,
        "snippet": f"Session title matched: {session_meta.get('title') or title_query}",
        "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or messages[:3])],
        "messages": [_shape_message(m, anchor_id=anchor_id) for m in (view.get("window") or messages[:5])],
        "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or messages[-3:])],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", max(len(messages) - 5, 0)),
        "_lineage_root": lineage_root,
    }
    if lineage_root and lineage_root != session_id:
        entry["parent_session_id"] = lineage_root
    return entry


def _discover(
    db,
    query: str,
    role_filter: Optional[List[str]],
    limit: int,
    sort: Optional[str],
    current_session_id: str = None,
) -> str:
    """Discovery shape: FTS5 + anchored window + bookends per hit. Single call."""
    role_list = role_filter if role_filter else ["user", "assistant"]
    current_lineage_root = _resolve_to_parent(db, current_session_id) if current_session_id else None
    title_result = _title_match_result(db, query, current_lineage_root)

    try:
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=_DISCOVER_SCAN_LIMIT,  # widen so dedup-by-lineage can find
            # distinct sessions AND so interactive matches buried under a wall
            # of cron rows are still in hand for the demotion pass below.
            offset=0,
            sort=sort,
        )
    except Exception as e:
        logging.error("FTS5 search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {e}", success=False)

    # 在去重之前，将自动化任务（Cron）行数据的优先级降至交互式会话之下，
    # 以防止高容量的 Cron 语料库挤占前 `limit` 条结果中的用户自有会话（#19434）。
    # 该逻辑保持稳定 —— 能够保留每种分类内部的 BM25 算法及近效性（Recency）排序。
    raw_results = _order_for_recall(raw_results)

    if not raw_results and not title_result:
        return json.dumps({
            "success": True,
            "mode": "discover",
            "query": query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        }, ensure_ascii=False)

    # 按谱系（Lineage）去重。
    # 在保留的行上保留原始所属的 session_id ——
    # 只有该 ID 才能与 FTS5 匹配点 ID 有效配对，用于构建锚点窗口。
    # 当 parent_session_id 不同时，会单独予以暴露。
    seen_sessions = {}
    results = []

    if title_result:
        title_lineage = title_result.pop("_lineage_root", None)
        if title_lineage:
            seen_sessions[title_lineage] = {"_title_only": True}
        results.append(title_result)

    for r in raw_results:
        if len(seen_sessions) >= limit:
            break
        raw_sid = r["session_id"]
        resolved_sid = _resolve_to_parent(db, raw_sid)
        # Skip the current session lineage
        if current_lineage_root and resolved_sid == current_lineage_root:
            continue
        if current_session_id and raw_sid == current_session_id:
            continue
        if resolved_sid not in seen_sessions:
            row = dict(r)
            row["_lineage_root"] = resolved_sid
            seen_sessions[resolved_sid] = row
        if len(seen_sessions) >= limit:
            break

    for lineage_root, match_info in seen_sessions.items():
        if match_info.get("_title_only"):
            continue
        hit_sid = match_info.get("session_id") or lineage_root
        msg_id = match_info.get("id")
        try:
            view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
        except Exception as e:
            logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
            continue

        try:
            session_meta = db.get_session(lineage_root) or {}
        except Exception:
            session_meta = {}

        entry = {
            "session_id": hit_sid,
            "when": _format_timestamp(
                session_meta.get("started_at") or match_info.get("session_started")
            ),
            "source": session_meta.get("source") or match_info.get("source", "unknown"),
            "model": session_meta.get("model") or match_info.get("model") or "unknown",
            "title": session_meta.get("title") or None,
            "matched_role": match_info.get("role"),
            "match_message_id": msg_id,
            "snippet": match_info.get("snippet") or "",
            "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or [])],
            "messages": [_shape_message(m, anchor_id=msg_id) for m in (view.get("window") or [])],
            "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or [])],
            "messages_before": view.get("messages_before", 0),
            "messages_after": view.get("messages_after", 0),
        }
        if lineage_root and lineage_root != hit_sid:
            entry["parent_session_id"] = lineage_root
        results.append(entry)

    return json.dumps({
        "success": True,
        "mode": "discover",
        "query": query,
        "results": results,
        "count": len(results),
        "sessions_searched": len(seen_sessions),
    }, ensure_ascii=False)


def session_search(
    query: str = "",
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
    # Scroll shape
    session_id: str = None,
    around_message_id: int = None,
    window: int = 5,
    # Discovery shape
    sort: str = None,
    # Cross-profile (any shape)
    profile: str = None,
) -> str:
    """单一调用的工具。模式根据传入的参数推断。

    探索模式（Discovery）：传入 ``query``。
    滚动模式（Scroll）：   传入 ``session_id`` + ``around_message_id``。
    读取模式（Read）：     仅传入 ``session_id``（无锚点消息）—— 导出整个会话。
    浏览模式（Browse）：   不传任何参数。

    传入 ``profile`` 可读取其他配置文件的会话（例如用于解析 ``@session:<profile>/<id>`` 链接）。
    当设置了锚点消息时，滚动模式优先于读取/探索模式 —— 这意味着 Agent 请求获取特定的消息切片。
    """
    if db is None:
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception:
            logging.debug("SessionDB unavailable for session_search", exc_info=True)
            from hermes_state import format_session_db_unavailable
            return tool_error(format_session_db_unavailable(), success=False)

    # 规范化传入 session_id 的原始 `@session:<profile>/<id>` 链接值。
    # Session ID 本身绝不包含 "/"，因此斜杠明确意味着 profile/id 格式 ——
    # 始终从 ID 中剥离该前缀，并且仅在未显式传递 profile 参数时，
    # 才采用链接中嵌入的 profile。
    # 这能处理模型可能发送的每种排列组合情况（如将完整值作为 ID 传入，附带或不附带独立的 profile=）。
    if isinstance(session_id, str) and "/" in session_id:
        emb_profile, _, emb_id = session_id.partition("/")
        if emb_id:
            session_id = emb_id
            if emb_profile and (profile is None or not str(profile).strip()):
                profile = emb_profile

    # 跨配置文件（Profile）读取：针对下方所有调用模式，
    # 均切换为指定 Profile 的数据库（只读）。
    # 当前会话的谱系防护机制在跨 Profile 时不再适用，
    # 但由于它们所依据的 ID 彼此不会冲突，因此这些机制会保持休眠（不生效）状态。
    if profile is not None and str(profile).strip():
        try:
            profile_db = _resolve_profile_db(profile)
        except Exception as e:
            return tool_error(f"profile '{profile}': {e}", success=False)
        if profile_db is not None:
            db = profile_db
            current_session_id = None

    # 滚动模式（Scroll shape）优先级最高 —— 显式指定的锚点消息高于任何查询词。
    if (isinstance(session_id, str) and session_id.strip()) and around_message_id is not None:
        return _scroll(
            db=db,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            current_session_id=current_session_id,
        )

    # 读取模式（Read shape）：传入了 session_id 但无锚点消息（anchor） → 导出整个会话。
    if isinstance(session_id, str) and session_id.strip():
        sid = session_id.strip()
        result = _read_session(db, sid)
        if json.loads(result).get("success"):
            return result

        # 在目标配置文件（Profile）中未命中 —— 模型可能从链接中遗漏了所属的 Profile。
        # 扫描每一个 Profile 并从其所在位置进行读取，
        # 同时标记出找到该 Session 时所匹配的 Profile。
        located, owner = _locate_session_db(sid)
        if located is not None:
            try:
                found = json.loads(_read_session(located, sid))
            finally:
                located.close()
            if found.get("success"):
                found["profile"] = owner
                return json.dumps(found, ensure_ascii=False)
        return result

    # Limit clamp [1, 10]
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 10))

    # Browse shape: no query → recent sessions.
    if not query or not isinstance(query, str) or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    # Parse role_filter
    role_list: Optional[List[str]] = None
    if isinstance(role_filter, str) and role_filter.strip():
        role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

    # Normalise sort
    sort_norm: Optional[str] = None
    if isinstance(sort, str):
        candidate = sort.strip().lower()
        if candidate in ("newest", "oldest"):
            sort_norm = candidate

    return _discover(
        db=db,
        query=query.strip(),
        role_filter=role_list,
        limit=limit,
        sort=sort_norm,
        current_session_id=current_session_id,
    )


def check_session_search_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False

# {
#     "name": "session_search",
#     "description": (
#         "搜索存储在本地会话数据库中的历史会话，或在指定会话内滚动浏览。\n"
#         "基于 SQLite 消息存储库的 FTS5 全文检索。无 LLM 调用 ——\n"
#         "每种调用形式均直接返回数据库中的实际消息内容。\n\n"
#         "源数据优先限制（SOURCE-FIRST LIMIT）\n\n"
#         "  本工具仅搜索 Hermes 的对话历史记录。它不能作为外部数据源当前内容的证据。\n"
#         "  如果用户提供了直接的数据源（如 URL、电话号码/联系人、应用/线程、文件路径、\n"
#         "  账号、网站或在线系统），在可行的情况下，请优先（或替代 session_search）\n"
#         "  检查该原始数据源。请将 session_search 用作了解历史对话的辅助上下文，\n"
#         "  而非证明数据源当前内容的直接依据。\n"
#         "  如果原始数据源无法访问，请在降级使用历史会话前说明情况及原因。\n"
#         "  当用户已提供直接数据源时，切勿仅凭 session_search 的结果就得出\n"
#         "  “未找到”或“无历史来往”的结论。\n\n"
#         "四种调用模式（FOUR CALLING SHAPES）\n\n"
#         "  1) 探索模式（DISCOVERY）—— 传入 `query`：\n"
#         "     session_search(query=\"auth refactor\", limit=3)\n"
#         "     运行 FTS5 搜索，按会话谱系去重，返回前 N 个匹配的会话。每个结果包含：\n"
#         "       - session_id、title、when、source\n"
#         "       - snippet: FTS5 高亮匹配的摘要片段\n"
#         "       - bookend_start: 会话的前 3 条 user+assistant 消息（目标/开端）\n"
#         "       - messages: FTS5 匹配点前后 ±5 条消息，并标记锚点消息（上下文中的命中点）\n"
#         "       - bookend_end: 会话的后 3 条 user+assistant 消息（结果/决策）\n"
#         "       - match_message_id、messages_before、messages_after\n"
#         "     首尾两端（Bookends）与消息窗口结合，无需调取完整转录即可重建“目标 → 匹配 → 决策”过程。\n\n"
#         "  2) 滚动模式（SCROLL）—— 传入 `session_id` + `around_message_id`：\n"
#         "     session_search(session_id=\"...\", around_message_id=12345, window=10)\n"
#         "     返回以锚点消息为中心、前后各 `window` 条消息的窗口。不使用 FTS5，无首尾片段，仅返回指定切片。\n"
#         "     用于探索调用后，当你需要比默认 ±5 条窗口更丰富的上下文时。\n"
#         "       - 向后滚动：将 messages[-1].id 作为 around_message_id 传入。\n"
#         "       - 向前滚动：将 messages[0].id 作为 around_message_id 传入。\n"
#         "       - 边界消息会在两个窗口中同时出现 —— 用作定位标记。\n"
#         "       - 当 messages_before 或 messages_after 小于 window 时，说明已触及会话开头或结尾。\n\n"
#         "  3) 读取模式（READ）—— 仅传入 `session_id`（无 around_message_id）：\n"
#         "     session_search(session_id=\"...\", profile=\"work\")\n"
#         "     根据 ID 导出整个会话（内容较长时返回前 20 条 + 后 10 条消息）。\n"
#         "     用于解析用户发在聊天中的 `@session:<profile>/<id>` 链接：\n"
#         "     按 `/` 切分出 profile 和 id，然后调用 session_search(session_id=id, profile=profile)。\n\n"
#         "  4) 浏览模式（BROWSE）—— 无参数：\n"
#         "     session_search()\n"
#         "     按时间顺序返回最近的会话：包含标题、预览和时间戳。\n"
#         "     当用户询问“我之前在忙什么”但未指定具体主题时使用。\n\n"
#         "FTS5 语法（FTS5 SYNTAX）\n\n"
#         "  默认使用 AND —— 多词查询需要包含所有词。显式使用 OR 可扩大检索范围（如 `alpha OR beta OR gamma`），\n"
#         "  精确匹配使用双引号（如 `\"docker networking\"`），逻辑非（如 `python NOT java`），\n"
#         "  或前缀通配符（如 `deploy*`）。\n\n"
#         "使用时机（WHEN TO USE）\n\n"
#         "  当遇到关于 Hermes 对话历史本身的问题时使用，例如“关于 X 我们之前做了什么”、“Y 进展到哪一步了”\n"
#         "  或“找到涉及 Z 的那个会话”。如果用户提供了直接的数据源标识符，在可行时应先检查该数据源；\n"
#         "  session_search 随后可用于补充历史上下文。会话数据库记录了何时说过什么；\n"
#         "  外部工具则展示当前的数据源/系统状态。"
#     ),
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": (
#                     "搜索查询词（探索模式）。用于在历史会话中查找的关键字、短语或布尔表达式。\n"
#                     "留空则用于浏览最近会话。当设置了 session_id + around_message_id 时（滚动模式），该参数会被忽略。"
#                 ),
#             },
#             "limit": {
#                 "type": "integer",
#                 "description": (
#                     "仅适用于探索模式。返回的最大会话数（默认为 3，最大为 10）。\n"
#                     "当主题可能跨越多个会话，且你想挑选合适的会话深入滚动浏览时，可调高至 5–10。"
#                 ),
#                 "default": 3,
#             },
#             "sort": {
#                 "type": "string",
#                 "enum": ["newest", "oldest"],
#                 "description": (
#                     "仅适用于探索模式。在 FTS5 相关性排序的基础上增加时间偏好。\n"
#                     "留空保持仅按相关性排序（适合探索性回顾 —— “关于 X 我们了解什么”）。\n"
#                     "设置 'newest' 用于偏向近期的问题（“X 进展到哪一步了”）。\n"
#                     "设置 'oldest' 用于偏向追根溯源的问题（“X 是怎么开始的”）。\n"
#                     "在滚动模式和浏览模式下忽略该参数。"
#                 ),
#             },
#             "session_id": {
#                 "type": "string",
#                 "description": (
#                     "滚动模式。要在内部读取的会话 ID。使用先前探索调用返回的 session_id。\n"
#                     "必须与 around_message_id 配对使用。"
#                 ),
#             },
#             "around_message_id": {
#                 "type": "integer",
#                 "description": (
#                     "滚动模式。作为窗口中心的消息 ID。可使用探索结果中的 match_message_id，\n"
#                     "或先前窗口中看到的任何 ID。向后滚动传入上一窗口最后一条消息的 ID；\n"
#                     "向前滚动传入第一条消息的 ID。"
#                 ),
#             },
#             "window": {
#                 "type": "integer",
#                 "description": (
#                     "仅适用于滚动模式。锚点消息两侧要返回的消息数量（锚点本身始终包含在内）。\n"
#                     "限制在 [1, 20] 范围内。默认为 5。"
#                 ),
#                 "default": 5,
#             },
#             "role_filter": {
#                 "type": "string",
#                 "description": (
#                     "可选。要包含的以逗号分隔的角色列表。探索模式默认为 'user,assistant'（工具输出通常为噪音）。\n"
#                     "传入 'user,assistant,tool' 以包含工具输出（用于调试工具行为），或传入 'tool' 仅搜索工具输出。"
#                 ),
#             },
#             "profile": {
#                 "type": "string",
#                 "description": (
#                     "可选。从另一个 Hermes 配置文件（Profile）的数据库中读取会话（只读）。\n"
#                     "解析 `@session:<profile>/<id>` 链接时使用：将配置部分传给此处，将 ID 部分传给 session_id。\n"
#                     "留空则使用当前配置文件。"
#                 ),
#             },
#         },
#         "required": [],
#     },
# }
SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past sessions stored in the local session DB, or scroll inside one. "
        "FTS5-backed retrieval over the SQLite message store. No LLM calls — every "
        "shape returns actual messages from the DB.\n\n"
        "SOURCE-FIRST LIMIT\n\n"
        "  This tool searches Hermes conversation history only. It is not evidence "
        "about the current contents of external sources. If the user provided a "
        "direct source such as a URL, phone number/contact, app/thread, file path, "
        "account, website, or live system, inspect that original source before or "
        "instead of session_search when accessible. Use session_search as secondary "
        "context for what was previously said, not as primary proof of what the "
        "source currently contains. If the original source is inaccessible, say so "
        "and why before falling back to session history. Do not conclude 'not found' "
        "or 'no prior correspondence' from session_search alone when a direct source "
        "was provided.\n\n"
        "FOUR CALLING SHAPES\n\n"
        "  1) DISCOVERY — pass `query`:\n"
        "     session_search(query=\"auth refactor\", limit=3)\n"
        "     Runs FTS5, dedupes hits by session lineage, returns the top N sessions. "
        "Each result carries:\n"
        "       - session_id, title, when, source\n"
        "       - snippet: FTS5-highlighted match excerpt\n"
        "       - bookend_start: first 3 user+assistant messages of the session "
        "(the goal / kickoff)\n"
        "       - messages: ±5 messages around the FTS5 match, with the anchor message "
        "flagged (the hit in context)\n"
        "       - bookend_end: last 3 user+assistant messages of the session "
        "(the resolution / decisions)\n"
        "       - match_message_id, messages_before, messages_after\n"
        "     Bookends + window together let you reconstruct goal → match → resolution "
        "without paying for the whole transcript.\n\n"
        "  2) SCROLL — pass `session_id` + `around_message_id`:\n"
        "     session_search(session_id=\"...\", around_message_id=12345, window=10)\n"
        "     Returns a window of ±`window` messages centered on the anchor. No FTS5, "
        "no bookends — just the slice. Use after a discovery call when you need more "
        "context than the ±5 default window.\n"
        "       - To scroll FORWARD: pass messages[-1].id back as around_message_id.\n"
        "       - To scroll BACKWARD: pass messages[0].id back as around_message_id.\n"
        "       - The boundary message appears in both windows — orientation marker.\n"
        "       - When messages_before or messages_after is < window, you're at the "
        "start or end of the session.\n\n"
        "  3) READ — pass `session_id` only (no around_message_id):\n"
        "     session_search(session_id=\"...\", profile=\"work\")\n"
        "     Dumps the whole session by id (first 20 + last 10 messages when "
        "large). This is how you resolve an `@session:<profile>/<id>` link the "
        "user dropped into the chat: split the value on `/` into profile + id "
        "and call session_search(session_id=id, profile=profile).\n\n"
        "  4) BROWSE — no args:\n"
        "     session_search()\n"
        "     Returns recent sessions chronologically: titles, previews, timestamps. "
        "Use when the user asks \"what was I working on\" without naming a topic.\n\n"
        "FTS5 SYNTAX\n\n"
        "  AND is the default — multi-word queries require all terms. Use OR explicitly "
        "for broader recall (`alpha OR beta OR gamma`), quoted phrases for exact match "
        "(`\"docker networking\"`), boolean (`python NOT java`), or prefix wildcards "
        "(`deploy*`).\n\n"
        "WHEN TO USE\n\n"
        "  Reach for this on questions about Hermes conversation history itself, such "
        "as \"what did we do about X\", \"where did we leave Y\", or \"find the "
        "session where Z\". If the user provided a direct source identifier, inspect "
        "that source first when accessible; session_search can then supply historical "
        "context. The session DB carries what was said when; external tools show "
        "current source/world state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (discovery shape). Keywords, phrases, or boolean "
                    "expressions to find in past sessions. Omit to browse recent "
                    "sessions. Ignored when session_id + around_message_id are set "
                    "(scroll shape)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Discovery shape only. Max sessions to return (default 3, max 10). "
                    "Bump to 5–10 when the topic likely spans several sessions and you "
                    "want to pick the right one to scroll into."
                ),
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": (
                    "Discovery shape only. Temporal bias on top of FTS5 ranking. Omit "
                    "to keep relevance-only ordering (suitable for exploratory recall — "
                    "\"what do we know about X\"). Set 'newest' for recency-shaped "
                    "questions (\"where did we leave X\"). Set 'oldest' for "
                    "origin-shaped questions (\"how did X start\"). Ignored in scroll "
                    "and browse shapes."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Scroll shape. Session to read inside. Use the session_id returned "
                    "from a prior discovery call. Must be paired with "
                    "around_message_id."
                ),
            },
            "around_message_id": {
                "type": "integer",
                "description": (
                    "Scroll shape. Message id to center the window on. From a discovery "
                    "result use match_message_id, or any id seen in a prior window. To "
                    "scroll forward pass the last window message's id; to scroll "
                    "backward pass the first."
                ),
            },
            "window": {
                "type": "integer",
                "description": (
                    "Scroll shape only. Messages to return on each side of the anchor "
                    "(anchor itself always included). Clamped to [1, 20]. Default 5."
                ),
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated roles to include. Discovery defaults to "
                    "'user,assistant' (tool output is usually noise). Pass "
                    "'user,assistant,tool' to include tool output (debugging tool "
                    "behaviour) or 'tool' to search tool output only."
                ),
            },
            "profile": {
                "type": "string",
                "description": (
                    "Optional. Read sessions from another Hermes profile's database "
                    "(read-only). Use when resolving an `@session:<profile>/<id>` link: "
                    "pass the profile segment here with session_id as the id segment. "
                    "Omit to use the current profile."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        profile=args.get("profile"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
