#!/usr/bin/env python3
"""
Fuzzy Matching Module for File Operations

Implements a multi-strategy matching chain to robustly find and replace text,
accommodating variations in whitespace, indentation, and escaping common
in LLM-generated code.

The 9-strategy chain (inspired by OpenCode), tried in order:
1. Exact match - Direct string comparison
2. Line-trimmed - Strip leading/trailing whitespace per line
3. Whitespace normalized - Collapse multiple spaces/tabs to single space
4. Indentation flexible - Ignore indentation differences entirely
5. Escape normalized - Convert \\n literals to actual newlines
6. Trimmed boundary - Trim first/last line whitespace only
7. Block anchor - Match first+last lines, use similarity for middle
8. Context-aware - 50% line similarity threshold

Multi-occurrence matching is handled via the replace_all flag.

Usage:
    from tools.fuzzy_match import fuzzy_find_and_replace
    
    new_content, match_count, strategy, error = fuzzy_find_and_replace(
        content="def foo():\\n    pass",
        old_string="def foo():",
        new_string="def bar():",
        replace_all=False
    )
"""

import re
from typing import Tuple, Optional, List, Callable
from difflib import SequenceMatcher

UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',  # smart double quotes
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u2014": "--", "\u2013": "-", # em/en dashes
    "\u2026": "...", "\u00a0": " ", # ellipsis and non-breaking space
}

def _unicode_normalize(text: str) -> str:
    """Normalizes Unicode characters to their standard ASCII equivalents."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


def fuzzy_find_and_replace(content: str, old_string: str, new_string: str,
                            replace_all: bool = False) -> Tuple[str, int, Optional[str], Optional[str]]:
    """
    使用一连串模糊程度递增的匹配策略进行查找与替换。

    参数：
        content: 拟在其中进行搜索的文件内容
        old_string: 待查找的文本
        new_string: 替换后的文本
        replace_all: 若为 True，则替换所有匹配项；若为 False，则要求匹配项必须唯一

    返回：
        元组 (new_content, match_count, strategy_name, error_message)
        - 成功时返回：(修改后的内容, 替换次数, 所用策略名称, None)
        - 失败时返回：(原始内容, 0, None, 错误描述信息)
    """
    if not old_string:
        return content, 0, None, "old_string cannot be empty"

    if old_string == new_string:
        return content, 0, None, "old_string and new_string are identical"

    # Try each matching strategy in order
    strategies: List[Tuple[str, Callable]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("unicode_normalized", _strategy_unicode_normalized),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, old_string)

        if matches:
            # Found matches with this strategy
            if len(matches) > 1 and not replace_all:
                return content, 0, None, (
                    f"Found {len(matches)} matches for old_string. "
                    f"Provide more context to make it unique, or use replace_all=True."
                )

            # 转义漂移防护机制：当匹配到的策略不是 `exact`（精确匹配）时，
            # 说明我们是通过某种形式的标准化处理才完成匹配的。
            # 如果 new_string 中包含了 shell 或 JSON 风格的转义序列（如 \' 或 \"），
            # 导致它们会被原封不动地字面写入文件中，但文件中被匹配到的区域
            # 本身却没有这类转义序列，这几乎可以肯定是由工具调用序列化漂移造成的——
            # 即模型输入了单引号/双引号，而传输层引入了多余的反斜杠。
            # 此时直接按原样写入 new_string 会损坏文件。
            # 因此抛出明确的提示信息予以拦截，让模型重新读取并重试，
            # 以防止调用方在无感知的情况下将垃圾数据持久化落盘。
            if strategy_name != "exact":
                drift_err = _detect_escape_drift(content, matches, old_string, new_string)
                if drift_err:
                    return content, 0, None, drift_err

            # 执行替换。当匹配策略非 `exact` 时，
            # 文件的缩进可能与 LLM 在 old_string/new_string 中发送的缩进不一致——
            # 例如：LLM 使用了 2 空格缩进，而文件使用的是 4 空格。
            # 根据缩进差值（indentation delta）平移 new_string，
            # 使替换内容符合文件实际的缩进模式。
            #
            # LLM 经常会将 JSON 工具调用（tool-call）参数中的制表符（tabs）和回车符（carriage returns）
            # 序列化为由两个字符组成的序列 ``\t`` 和 ``\r``（反斜杠 + 字母），而不是实际的控制字节。
            # 如果我们原样写入 new_string，当周围代码使用真实制表符时，
            # 文件中最终会留下字面量的反斜杠序列。
            #
            # 策略：仅当文件的匹配区域*确实包含*对应的真实控制字符时才进行反转义（unescape）。
            # 这与 ``_detect_escape_drift`` 中基于区域的启发式规则保持一致，
            # 并且能够保留对字面量双字符字符串 ``"\t"`` 的合法写入
            # （例如：修补源码本身就包含制表符字符串字面量的 Python 代码）——
            # 这些文件在匹配区域中包含的是“反斜杠+t”，而不是真实的制表符，因此我们不对 new_string 做任何修改。
            #
            # 特意排除了 ``\n``：换行符可以通过 JSON 正确序列化，
            # 而重写“反斜杠-n”破坏源码常量中转义序列的概率，远大于其提供帮助的概率。
            effective_new = _maybe_unescape_new_string(
                new_string, content, matches,
            )
            # Unicode 保留防护机制：
            # 当策略 7（unicode_normalized）匹配成功时，
            # 说明文件中包含 Unicode 字符（如破折号、弯引号、省略号等），
            # 但来自 LLM 的 old_string/new_string 却是对应的 ASCII 替代字符。
            #
            # 如果原样写入 new_string，会静默损坏文件中的 Unicode 字符——
            # 例如破折号变成了两个连字符，弯引号变成了直引号。
            #
            # 因此，需要将替换内容与文件中实际的 Unicode 字符进行对齐，
            # 从而仅应用 LLM 预期的修改，
            # 并让未修改的部分继续保留其原始字符。
            if strategy_name == "unicode_normalized":
                effective_new = _preserve_unicode_in_replacement(
                    content, matches, old_string, effective_new,
                )
            new_content = _apply_replacements(
                content, matches, effective_new,
                old_string=old_string if strategy_name != "exact" else None,
            )
            return new_content, len(matches), strategy_name, None

    # No strategy found a match
    return content, 0, None, "Could not find a match for old_string in the file"


def _detect_escape_drift(content: str, matches: List[Tuple[int, int]],
                         old_string: str, new_string: str) -> Optional[str]:
    """检测 new_string 中因工具调用转义漂移（escape-drift）产生的残留字符。

    查找在 old_string 和 new_string 中同时存在
    （即模型将其作为打算保留的“上下文”进行了复制粘贴）、
    但在文件匹配区域中并不存在的 ``\\'`` 或 ``\\"`` 序列。
    这种特征表明传输层在单引号或双引号周围插入了伪造的 shell 风格转义字符——
    如果原封不动地写入 new_string，会导致将 ``\\'`` 字面量直接插入到源代码中。

    如果检测到转义漂移，则返回错误字符串，否则返回 None。
    """
    # 快速预检查：除非 new_string 确实包含可疑的转义序列，
    # 否则直接跳过处理。这保证了所有常见且正确的场景不会产生额外开销。
    if "\\'" not in new_string and '\\"' not in new_string:
        return None

    # 聚合文件中被匹配到的区域——即 new_string 将要替换的部分。
    # 如果可疑的转义序列在这些区域中已经存在，
    # 说明模型确实是在保留它们（这对某些语言或转义字符串是合理的）；
    # 此时接受该补丁（patch）。
    matched_regions = "".join(content[start:end] for start, end in matches)

    for suspect in ("\\'", '\\"'):
        if suspect in new_string and suspect in old_string and suspect not in matched_regions:
            plain = suspect[1]  # "'" or '"'
            # return (
            #     f"检测到转义漂移（Escape-drift）：old_string 和 new_string 包含 "
            #     f"字面量序列 {suspect!r}，但文件中被匹配的区域并不存在该序列。\n"
            #     f"这几乎总是由于工具调用序列化过程产生的残留问题——\n"
            #     f"单引号或双引号前被误加了多余的反斜杠。\n"
            #     f"请使用 read_file 重新读取该文件，并在传入 old_string/new_string 时\n"
            #     f"不要对 {plain!r} 字符进行反斜杠转义。"
            # )
            return (
                f"Escape-drift detected: old_string and new_string contain "
                f"the literal sequence {suspect!r} but the matched region of "
                f"the file does not. This is almost always a tool-call "
                f"serialization artifact where an apostrophe or quote got "
                f"prefixed with a spurious backslash. Re-read the file with "
                f"read_file and pass old_string/new_string without "
                f"backslash-escaping {plain!r} characters."
            )
    return None


def _leading_whitespace(line: str) -> str:
    """Return the leading whitespace prefix of a line (spaces/tabs)."""
    i = 0
    while i < len(line) and line[i] in (" ", "\t"):
        i += 1
    return line[:i]


def _first_meaningful_line(text: str) -> Optional[str]:
    """Return the first line of ``text`` that has any non-whitespace content.

    Returns ``None`` if no such line exists (text is empty or all whitespace).
    """
    for line in text.split("\n"):
        if line.strip():
            return line
    return None


def _reindent_replacement(file_region: str, old_string: str, new_string: str) -> str:
    """调整 ``new_string``，使其缩进与 ``file_region`` 保持一致。

    用于非精确的模糊匹配（non-exact fuzzy match）之后：
    LLM 发送的 old_string 和 new_string 缩进可能与文件实际缩进不一致
    （例如：工具参数中使用 2 空格缩进，而磁盘文件上是 4 空格缩进）。
    尽管模糊策略成功匹配上了，但如果原样写入 ``new_string``，
    会破坏文件的缩进。

    实现方法：

    1. 对于 ``new_string`` 中的每一非空行，计算其*相对于* ``old_string`` 中
       最浅非空行（即 LLM 的基准缩进）的相对缩进。
    2. 将该相对缩进锚定到文件实际的基准缩进上
       （即 file_region 中首个非空行的前导空白字符）。
    3. 将每个非空行重新生成为 ``file_base + (line_indent - llm_base)``。

    空行以及缩进浅于 LLM 基准缩进的行，直接锚定到文件的基准缩进上。

    无需处理（原样返回 ``new_string``）的情况：
    - file_region 或 old_string 中没有有效的行
    - LLM 的基准缩进等于文件的基准缩进
    - new_string 为空
    """
    if not new_string:
        return new_string

    old_first = _first_meaningful_line(old_string)
    file_first = _first_meaningful_line(file_region)
    if old_first is None or file_first is None:
        return new_string

    old_indent = _leading_whitespace(old_first)
    file_indent = _leading_whitespace(file_first)

    if old_indent == file_indent:
        return new_string

    # 重新缩进 new_string 的每一行。策略：用文件的基准缩进前缀
    # 替换 LLM 的基准缩进前缀，同时保留 LLM 在此基础上增加的任何
    # 额外缩进。这与 Roo Code 所采用的方法一致
    # （multi-search-replace.ts:466-500）。它能够保留 LLM 预期的
    # 行间*相对*嵌套关系，同时将其锚定到文件实际的缩进风格上。
    out_lines: List[str] = []
    for line in new_string.split("\n"):
        if not line.strip():
            # 空行：保留空白字符不做修改。
            out_lines.append(line)
            continue
        line_indent = _leading_whitespace(line)
        if line_indent.startswith(old_indent):
            # 常见情况：行内包含 LLM 的基准缩进（可能包含额外缩进）。
            # 将基准前缀替换为文件的基准前缀。
            remainder = line[len(old_indent):]
            out_lines.append(file_indent + remainder)
        else:
            # 该行缩进浅于 LLM 的基准缩进——
            # 例如 new_string 开头的取消缩进（dedent）。
            # 将其直接锚定到文件的基准缩进上。
            out_lines.append(file_indent + line.lstrip(" \t"))
    return "\n".join(out_lines)


def _maybe_unescape_new_string(new_string: str,
                               content: str,
                               matches: List[Tuple[int, int]]) -> str:
    """条件性地取消 new_string 中 ``\t``/``\r`` 的转义。

    在 JSON 工具调用参数中，LLM 经常在原本想表达真实制表符（tab）或回车符（carriage-return）字节的地方，
    发送由两个字符组成的序列 ``\t``（反斜杠 + t）和 ``\r``（反斜杠 + r）。
    如果原样写入该字符串，会用字面的“反斜杠-字母”对损坏采用制表符缩进的文件。

    仅当*文件的匹配区域*实际上包含对应的控制字符时，才会对各个序列应用取消转义——
    也就是说，只有当我们要替换的文件区域包含真实的制表符字节时，我们才会将 ``\t`` 转换为制表符。
    对于合法包含字面量双字符字符串 ``"\t"`` 的文件
    （例如定义了 ``sep = "\t"`` 的 Python 源码行），其匹配区域包含的是反斜杠+t 而非真实制表符，
    因此我们会对 new_string 保持原样。

    特意排除了 ``\n``：换行符可以通过 JSON 正确序列化，
    而重写“反斜杠-n”破坏字符串字面量中转义序列的概率，远大于其提供帮助的概率。
    """
    # 快速预检 —— 除非 new_string 确实包含可疑序列之一，否则直接退出。
    # 保持常规情况下的零开销。
    if "\\t" not in new_string and "\\r" not in new_string:
        return new_string

    matched_regions = "".join(content[start:end] for start, end in matches)
    out = new_string
    if "\\t" in out and "\t" in matched_regions:
        out = out.replace("\\t", "\t")
    if "\\r" in out and "\r" in matched_regions:
        out = out.replace("\\r", "\r")
    return out


def _preserve_unicode_in_replacement(
    content: str, matches: List[Tuple[int, int]],
    old_string: str, new_string: str,
) -> str:
    """保留文件中替换字符串里的 Unicode 字符。

    当策略 7（unicode_normalized）匹配成功时，
    说明文件中包含 Unicode 字符（如破折号、弯引号、省略号、不换行空格等），
    但来自 LLM 的 old_string/new_string 却是对应的 ASCII 替代字符。
    如果原样写入 new_string，会静默损坏文件中的 Unicode 字符——
    例如破折号变成了两个连字符，弯引号变成了直引号。

    本函数通过对 old_string→new_string 进行差异对比（diff），
    并将实际的修改应用到文件的原始文本中，
    从而使替换内容与文件实际的 Unicode 字符保持对齐，
    并保留未修改部分的 Unicode 字符。
    """
    # 聚合匹配的文件区域
    file_region = "".join(content[start:end] for start, end in matches)

    # Normalize both for comparison
    norm_old = _unicode_normalize(old_string)
    norm_file = _unicode_normalize(file_region)

    # If the normalized forms don't match, the strategy shouldn't have
    # fired — fall back to direct replacement.
    if norm_old != norm_file:
        return new_string

    # 分别为 old_string 和 file_region 构建
    # 从标准化空间映射回原始空间的位置映射表。
    # UNICODE_MAP 的替换可能会展开字符（例如破折号 → '--'），
    # 因此标准化后的位置与原始位置之间并非 1:1 的映射关系。
    # 此处复用模块级的 _build_orig_to_norm_map，
    # 然后将其反转（与 _map_positions_norm_to_orig 中的反转逻辑相同），
    # 从而获得从标准化位置到原始位置（norm→orig）的查找表。
    file_orig_to_norm = _build_orig_to_norm_map(file_region)
    file_norm_to_orig: dict[int, int] = {}
    for orig_pos, np in enumerate(file_orig_to_norm[:-1]):
        if np not in file_norm_to_orig:
            file_norm_to_orig[np] = orig_pos

    # Diff norm_old → new_string to find the actual edits
    sm = SequenceMatcher(None, norm_old, new_string)
    opcodes = sm.get_opcodes()

    # Apply edits to file_region, preserving Unicode for unchanged spans
    result_parts: List[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Keep the original file_region text for this span
            orig_start = file_norm_to_orig.get(i1, 0)
            orig_end = orig_start
            while (
                orig_end < len(file_region)
                and file_orig_to_norm[orig_end] < i2
            ):
                orig_end += 1
            result_parts.append(file_region[orig_start:orig_end])
        elif tag == "replace":
            result_parts.append(new_string[j1:j2])
        elif tag == "delete":
            pass  # skip deleted portion
        elif tag == "insert":
            result_parts.append(new_string[j1:j2])

    return "".join(result_parts)


def _apply_replacements(content: str, matches: List[Tuple[int, int]],
                        new_string: str, old_string: Optional[str] = None) -> str:
    """
    在指定位置应用替换。

    参数:
        content: 原始内容
        matches: 待替换的 (start, end) 位置列表
        new_string: 替换文本
        old_string: 当不为 None 时，表示该匹配来自于非精确的模糊匹配策略；
            在进行替换前，``new_string`` 会被重新缩进，
            以匹配文件实际的缩进。

    返回:
        应用替换后的内容
    """
    # 按位置倒序（从大到小）对匹配项进行排序，以便从后往前进行替换
    # 这样可以保持前面匹配项的位置索引不受影响
    sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)

    result = content
    for start, end in sorted_matches:
        if old_string is not None:
            file_region = content[start:end]
            adjusted = _reindent_replacement(file_region, old_string, new_string)
        else:
            adjusted = new_string
        result = result[:start] + adjusted + result[end:]

    return result


# =============================================================================
# Matching Strategies
# =============================================================================

def _strategy_exact(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Strategy 1: Exact string match."""
    matches = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        # Advance past the whole match, not just one char, so self-overlapping
        # patterns (e.g. "aa" in "aaaa") produce non-overlapping spans matching
        # str.replace() semantics. Advancing by 1 yielded overlapping matches
        # that corrupt the file under replace_all=True (reverse-order apply on
        # stale offsets).
        start = pos + len(pattern)
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    策略 2：通过逐行去除首尾空格来进行匹配。

    在匹配前去除每行文本的前导与尾随空格。
    """
    # 通过去除每行的首尾空格来规范化模式 (pattern) 和内容 (content)
    pattern_lines = [line.strip() for line in pattern.split('\n')]
    pattern_normalized = '\n'.join(pattern_lines)
    
    content_lines = content.split('\n')
    content_normalized_lines = [line.strip() for line in content_lines]
    
    # Build mapping from normalized positions back to original positions
    return _find_normalized_matches(
        content, content_lines, content_normalized_lines,
        pattern, pattern_normalized
    )


def _strategy_whitespace_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 3: Collapse multiple whitespace to single space.
    """
    def normalize(s):
        # Collapse multiple spaces/tabs to single space, preserve newlines
        return re.sub(r'[ \t]+', ' ', s)
    
    pattern_normalized = normalize(pattern)
    content_normalized = normalize(content)
    
    # Find in normalized, map back to original
    matches_in_normalized = _strategy_exact(content_normalized, pattern_normalized)
    
    if not matches_in_normalized:
        return []
    
    # Map positions back to original content
    return _map_normalized_positions(content, content_normalized, matches_in_normalized)


def _strategy_indentation_flexible(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 4: Ignore indentation differences entirely.
    
    Strips all leading whitespace from lines before matching.
    """
    content_lines = content.split('\n')
    content_stripped_lines = [line.lstrip() for line in content_lines]
    pattern_lines = [line.lstrip() for line in pattern.split('\n')]
    
    return _find_normalized_matches(
        content, content_lines, content_stripped_lines,
        pattern, '\n'.join(pattern_lines)
    )


def _strategy_escape_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 5: Convert escape sequences to actual characters.
    
    Handles \\n -> newline, \\t -> tab, etc.
    """
    def unescape(s):
        # Convert common escape sequences
        return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    
    pattern_unescaped = unescape(pattern)
    
    if pattern_unescaped == pattern:
        # No escapes to convert, skip this strategy
        return []
    
    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 6: Trim whitespace from first and last lines only.
    
    Useful when the pattern boundaries have whitespace differences.
    """
    pattern_lines = pattern.split('\n')
    if not pattern_lines:
        return []
    
    # Trim only first and last lines
    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()
    
    modified_pattern = '\n'.join(pattern_lines)
    
    content_lines = content.split('\n')
    
    # Search through content for matching block
    matches = []
    pattern_line_count = len(pattern_lines)
    
    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i:i + pattern_line_count]
        
        # Trim first and last of this block
        check_lines = block_lines.copy()
        check_lines[0] = check_lines[0].strip()
        if len(check_lines) > 1:
            check_lines[-1] = check_lines[-1].strip()
        
        if '\n'.join(check_lines) == modified_pattern:
            # Found match - calculate original positions
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _build_orig_to_norm_map(original: str) -> List[int]:
    """构建一个将每个原始字符索引映射到其标准化索引的列表。

    由于 UNICODE_MAP 替换可能会展开字符（例如：破折号 → '--'，
    省略号 → '...'），因此标准化后的字符串可能会比原始字符串更长。
    通过该映射表，我们可以将标准化字符串中的位置
    转换回原始字符串中的对应位置。

    返回一个长度为 ``len(original) + 1`` 的列表；
    第 ``i`` 个条目代表字符 ``i`` 所映射到的标准化索引。
    """
    result: List[int] = []
    norm_pos = 0
    for char in original:
        result.append(norm_pos)
        repl = UNICODE_MAP.get(char)
        norm_pos += len(repl) if repl is not None else 1
    result.append(norm_pos)  # sentinel: one past the last character
    return result


def _map_positions_norm_to_orig(
    orig_to_norm: List[int],
    norm_matches: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Convert (start, end) positions in the normalised string to original positions."""
    # Invert the map: norm_pos -> first original position with that norm_pos
    norm_to_orig_start: dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm[:-1]):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos

    results: List[Tuple[int, int]] = []
    orig_len = len(orig_to_norm) - 1  # number of original characters

    for norm_start, norm_end in norm_matches:
        if norm_start not in norm_to_orig_start:
            continue
        orig_start = norm_to_orig_start[norm_start]

        # Walk forward until orig_to_norm[orig_end] >= norm_end
        orig_end = orig_start
        while orig_end < orig_len and orig_to_norm[orig_end] < norm_end:
            orig_end += 1

        results.append((orig_start, orig_end))

    return results


def _strategy_unicode_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Strategy 7: Unicode normalisation.

    Normalises smart quotes, em/en-dashes, ellipsis, and non-breaking spaces
    to their ASCII equivalents in both *content* and *pattern*, then runs
    exact and line_trimmed matching on the normalised copies.

    Positions are mapped back to the *original* string via
    ``_build_orig_to_norm_map`` — necessary because some UNICODE_MAP
    replacements expand a single character into multiple ASCII characters,
    making a naïve position copy incorrect.
    """
    # Normalize both sides. Either the content or the pattern (or both) may
    # carry unicode variants — e.g. content has an em-dash that should match
    # the LLM's ASCII '--', or vice-versa.  Skip only when neither changes.
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    if norm_content == content and norm_pattern == pattern:
        return []

    norm_matches = _strategy_exact(norm_content, norm_pattern)
    if not norm_matches:
        norm_matches = _strategy_line_trimmed(norm_content, norm_pattern)

    if not norm_matches:
        return []

    orig_to_norm = _build_orig_to_norm_map(content)
    return _map_positions_norm_to_orig(orig_to_norm, norm_matches)


def _strategy_block_anchor(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 8: Match by anchoring on first and last lines.
    Adjusted with permissive thresholds and unicode normalization.
    """
    # Normalize both strings for comparison while keeping original content for offset calculation
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    
    pattern_lines = norm_pattern.split('\n')
    if len(pattern_lines) < 2:
        return []
    
    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()
    
    # Use normalized lines for matching logic
    norm_content_lines = norm_content.split('\n')
    # BUT use original lines for calculating start/end positions to prevent index shift
    orig_content_lines = content.split('\n')
    
    pattern_line_count = len(pattern_lines)
    
    potential_matches = []
    for i in range(len(norm_content_lines) - pattern_line_count + 1):
        if (norm_content_lines[i].strip() == first_line and 
            norm_content_lines[i + pattern_line_count - 1].strip() == last_line):
            potential_matches.append(i)
            
    matches = []
    candidate_count = len(potential_matches)
    
    # Thresholding logic: 0.50 for unique matches, 0.70 for multiple candidates.
    # Previous values (0.10 / 0.30) were dangerously loose — a 10% middle-section
    # similarity could match completely unrelated blocks.
    threshold = 0.50 if candidate_count == 1 else 0.70

    for i in potential_matches:
        if pattern_line_count <= 2:
            similarity = 1.0
        else:
            # Compare normalized middle sections
            content_middle = '\n'.join(norm_content_lines[i+1:i+pattern_line_count-1])
            pattern_middle = '\n'.join(pattern_lines[1:-1])
            similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()
        
        if similarity >= threshold:
            # Calculate positions using ORIGINAL lines to ensure correct character offsets in the file
            start_pos, end_pos = _calculate_line_positions(
                orig_content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _strategy_context_aware(content: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Strategy 9: Line-by-line similarity with 50% threshold.
    
    Finds blocks where at least 50% of lines have high similarity.
    """
    pattern_lines = pattern.split('\n')
    content_lines = content.split('\n')
    
    if not pattern_lines:
        return []
    
    matches = []
    pattern_line_count = len(pattern_lines)
    
    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i:i + pattern_line_count]
        
        # Calculate line-by-line similarity
        high_similarity_count = 0
        for p_line, c_line in zip(pattern_lines, block_lines):
            sim = SequenceMatcher(None, p_line.strip(), c_line.strip()).ratio()
            if sim >= 0.80:
                high_similarity_count += 1
        
        # Need at least 50% of lines to have high similarity
        if high_similarity_count >= len(pattern_lines) * 0.5:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


# =============================================================================
# Helper Functions
# =============================================================================

def _calculate_line_positions(content_lines: List[str], start_line: int,
                              end_line: int, content_length: int) -> Tuple[int, int]:
    """Calculate start and end character positions from line indices.

    Args:
        content_lines: List of lines (without newlines)
        start_line: Starting line index (0-based)
        end_line: Ending line index (exclusive, 0-based)
        content_length: Total length of the original content string

    Returns:
        Tuple of (start_pos, end_pos) in the original content
    """
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    end_pos = min(content_length, end_pos)
    return start_pos, end_pos


def _find_normalized_matches(content: str, content_lines: List[str],
                              content_normalized_lines: List[str],
                              pattern: str, pattern_normalized: str) -> List[Tuple[int, int]]:
    """
    在规范化后的内容中查找匹配项，并映射回原始位置。

    参数：
        content: 原始内容字符串
        content_lines: 按行分割的原始内容
        content_normalized_lines: 规范化后的内容行
        pattern: 原始模式
        pattern_normalized: 规范化后的模式

    返回：
        原始内容中 (start, end) 起止位置的列表
    """
    pattern_norm_lines = pattern_normalized.split('\n')
    num_pattern_lines = len(pattern_norm_lines)
    
    matches = []
    
    for i in range(len(content_normalized_lines) - num_pattern_lines + 1):
        # Check if this block matches
        block = '\n'.join(content_normalized_lines[i:i + num_pattern_lines])
        
        if block == pattern_normalized:
            # Found a match - calculate original positions
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + num_pattern_lines, len(content)
            )
            matches.append((start_pos, end_pos))
    
    return matches


def _map_normalized_positions(original: str, normalized: str,
                               normalized_matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Map positions from normalized string back to original.
    
    This is a best-effort mapping that works for whitespace normalization.
    """
    if not normalized_matches:
        return []
    
    # Build character mapping from normalized to original
    orig_to_norm = []  # orig_to_norm[i] = position in normalized
    
    orig_idx = 0
    norm_idx = 0
    
    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in ' \t' and normalized[norm_idx] == ' ':
            # Original has space/tab, normalized collapsed to space
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            # Don't advance norm_idx yet - wait until all whitespace consumed
            if orig_idx < len(original) and original[orig_idx] not in ' \t':
                norm_idx += 1
        elif original[orig_idx] in ' \t':
            # Extra whitespace in original
            orig_to_norm.append(norm_idx)
            orig_idx += 1
        else:
            # Mismatch - shouldn't happen with our normalization
            orig_to_norm.append(norm_idx)
            orig_idx += 1
    
    # Fill remaining
    while orig_idx < len(original):
        orig_to_norm.append(len(normalized))
        orig_idx += 1
    
    # Reverse mapping: for each normalized position, find original range
    norm_to_orig_start = {}
    norm_to_orig_end = {}
    
    for orig_pos, norm_pos in enumerate(orig_to_norm):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos
        norm_to_orig_end[norm_pos] = orig_pos
    
    # Map matches
    original_matches = []
    for norm_start, norm_end in normalized_matches:
        # Find original start
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            # Find nearest
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)
        
        # Find original end
        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)
        
        # Expand to include trailing whitespace that was normalized,
        # but only when the normalized match itself ended with whitespace.
        # When the match ends with a non-space character, the first
        # whitespace in the original is a word boundary and must not be
        # consumed.  See https://github.com/NousResearch/hermes-agent/issues/52491
        if norm_end < len(normalized) and normalized[norm_end - 1] == ' ':
            while orig_end < len(original) and original[orig_end] in ' \t':
                orig_end += 1
        
        original_matches.append((orig_start, min(orig_end, len(original))))
    
    return original_matches


def find_closest_lines(old_string: str, content: str, context_lines: int = 2, max_results: int = 3) -> str:
    """Find lines in content most similar to old_string for "did you mean?" feedback.

    Returns a formatted string showing the closest matching lines with context,
    or empty string if no useful match is found.
    """
    if not old_string or not content:
        return ""

    old_lines = old_string.splitlines()
    content_lines = content.splitlines()

    if not old_lines or not content_lines:
        return ""

    # Use first line of old_string as anchor for search
    anchor = old_lines[0].strip()
    if not anchor:
        # Try second line if first is blank
        candidates = [l.strip() for l in old_lines if l.strip()]
        if not candidates:
            return ""
        anchor = candidates[0]

    # Score each line in content by similarity to anchor
    scored = []
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        if not stripped:
            continue
        ratio = SequenceMatcher(None, anchor, stripped).ratio()
        if ratio > 0.3:
            scored.append((ratio, i))

    if not scored:
        return ""

    # Take top matches
    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]

    parts = []
    seen_ranges = set()
    for _, line_idx in top:
        start = max(0, line_idx - context_lines)
        end = min(len(content_lines), line_idx + len(old_lines) + context_lines)
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        snippet = "\n".join(
            f"{start + j + 1:4d}| {content_lines[start + j]}"
            for j in range(end - start)
        )
        parts.append(snippet)

    if not parts:
        return ""

    return "\n---\n".join(parts)


def format_no_match_hint(error: Optional[str], match_count: int,
                         old_string: str, content: str) -> str:
    """Return a '\\n\\nDid you mean...' snippet for plain no-match errors.

    Gated so the hint only fires for actual "old_string not found" failures.
    Ambiguous-match ("Found N matches"), escape-drift, and identical-strings
    errors all have ``match_count == 0`` but a "did you mean?" snippet would
    be misleading — those failed for unrelated reasons.

    Returns an empty string when there's nothing useful to append.
    """
    if match_count != 0:
        return ""
    if not error or not error.startswith("Could not find"):
        return ""
    hint = find_closest_lines(old_string, content)
    if not hint:
        return ""
    return "\n\nDid you mean one of these sections?\n" + hint
