"""Per-session event sequencing + bounded replay for WS reconnects.

Every gateway event frame that flows through :func:`server.write_json` (and
therefore ``_emit``) is stamped with a per-session monotonic ``seq`` and
appended to a small ring buffer keyed by session id. A reconnecting client
calls the ``session.events.since`` RPC with its last observed seq; the server
replays everything newer from the buffer, then live events resume seamlessly.

Design constraints honored:
- stdio TUI path unaffected: frames gain a ``seq`` field only on event frames;
  Ink ignores unknown params keys.
- Thread safety: a single module lock guards counters + buffers; write_json
  already serializes per-transport writes, so stamping under the lock cannot
  reorder frames relative to each other.
- Memory bound: _REPLAY_BUFFER_MAX events / _REPLAY_SESSIONS_MAX sessions,
  oldest session evicted FIFO.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict, deque

# 用于重放协议的进程标识。序列号（Seq）计数器存在于进程内存中，
# 因此网关重启会静默将其重置为 1，而此时客户端仍保留着较高的水位线——
# 这会导致 events_since(sid, 97) 返回 [] 且 truncated=False，
# 从而使客户端误以为没有丢失任何数据
# （并且其陈旧的水位线会让未来的每一次重放都返回空）。
# epoch 机制可以让客户端检测到重启并重置其水位线。
_REPLAY_EPOCH = uuid.uuid4().hex

# 每个会话的重放环形缓冲区。一个长轮次对话会产生约数百个 token 事件；
# 该容量可覆盖数分钟的流式传输以及所有的控制事件。
_REPLAY_BUFFER_MAX = 512
# 记忆的独立会话上限。桌面端用户极少会同时开启超过十几个活跃对话。
_REPLAY_SESSIONS_MAX = 64

_replay_lock = threading.Lock()
# sid -> (seq, event_object) 双端队列（deque），
# 其中 event_object 为该帧的 ``params`` 字典
# （纯事件：包含 type/session_id/seq/payload）——
# 这正是客户端分发路径所消耗的精确数据格式。
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
_replay_next_seq: dict[str, int] = {}

def replay_epoch() -> str:
    """Opaque token identifying this server process's seq numbering."""
    return _REPLAY_EPOCH


def _stamp_event(obj: dict) -> None:
    """Stamp one outgoing event frame (mutates obj in place) and record it."""
    if obj.get("method") != "event":
        return
    params = obj.get("params")
    if not isinstance(params, dict):
        return
    sid = params.get("session_id") or ""
    if not sid:
        # Session-less global events (skin.changed etc.) are re-fetchable via
        # their own RPCs; no replay contract for them.
        return
    with _replay_lock:
        seq = _replay_next_seq.get(sid, 0) + 1
        _replay_next_seq[sid] = seq
        params["seq"] = seq
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = deque(maxlen=_REPLAY_BUFFER_MAX)
            _replay_buffers[sid] = buf
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                _oldest_sid, _oldest_buf = _replay_buffers.popitem(last=False)
                _replay_next_seq.pop(_oldest_sid, None)
        buf.append((seq, params))


def events_since(sid: str, last_seen: int) -> list[dict]:
    """按顺序返回 *sid* 对应且 seq > last_seen 的已记录事件对象（EVENT OBJECTS）。

    数据格式协议：每个元素都是该帧的 ``params`` 字典——
    即一个包含顶层 ``type`` / ``session_id`` / ``seq`` 的纯事件对象——
    因为这正是客户端分发路径所消耗的数据。
    如果在这里返回完整的 JSON-RPC 包装层，
    会导致所有重放的事件都无法通过客户端的 ``event.type`` 校验而被静默丢弃。
    """
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return []
        return [event for seq, event in buf if seq > last_seen]


def is_truncated(sid: str, last_seen: int) -> bool:
    """True when events between *last_seen* and the ring's oldest retained
    seq were evicted — the client must refetch history instead of trusting
    the replay to be gap-free."""
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return False
        return last_seen + 1 < buf[0][0]


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook."""
    with _replay_lock:
        _replay_buffers.clear()
        _replay_next_seq.clear()


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(b) for b in _replay_buffers.values()),
            "max_per_session": _REPLAY_BUFFER_MAX,
        }
