from jarv.auditor import _parse_response


def test_parse_response_accepts_strict_json():
    assert _parse_response('{"allow": true, "reason": "routine install"}') == (
        True,
        "routine install",
    )


def test_parse_response_accepts_json_with_surrounding_text():
    assert _parse_response('Sure.\n{"allow": true, "reason": "safe in context"}') == (
        True,
        "safe in context",
    )


def test_parse_response_accepts_json_after_powershell_block_text():
    assert _parse_response(
        'The command includes if (-not $?) { exit 1 }.\n'
        '{"allow": true, "reason": "only removes a temp file"}'
    ) == (
        True,
        "only removes a temp file",
    )


def test_parse_response_accepts_json_before_powershell_block_text():
    assert _parse_response(
        '{"allow": true, "reason": "safe tag recreation"}\n'
        "Note: cleanup uses if (-not $?) { exit 1 }."
    ) == (
        True,
        "safe tag recreation",
    )


def test_parse_response_skips_invalid_braces_before_json():
    assert _parse_response(
        'First brace is not JSON: { exit 1 }.\n'
        '{"allow": "false", "reason": "tag deletion needs review"}'
    ) == (
        False,
        "tag deletion needs review",
    )


def test_parse_response_accepts_fenced_json_with_surrounding_text():
    assert _parse_response(
        'Verdict:\n```json\n{"allow": "true", "reason": "temp file cleanup only"}\n```\nDone.'
    ) == (
        True,
        "temp file cleanup only",
    )


def test_parse_response_accepts_json_reason_containing_braces():
    assert _parse_response(
        '{"allow": true, "reason": "PowerShell block { exit 1 } is only error handling"}'
    ) == (
        True,
        "PowerShell block { exit 1 } is only error handling",
    )


def test_parse_response_ignores_json_without_allow_key():
    assert _parse_response(
        '{"reason": "missing verdict"}\n{"allow": true, "reason": "safe cleanup"}'
    ) == (
        True,
        "safe cleanup",
    )


def test_parse_response_rejects_unclear_json_allow_value():
    assert _parse_response('{"allow": "maybe", "reason": "ambiguous"}') == (
        False,
        "could not parse auditor response",
    )


def test_parse_response_accepts_allow_key_value_text():
    assert _parse_response("allow: true\nreason: version probe only") == (
        True,
        "version probe only",
    )


def test_parse_response_accepts_leading_allow_verdict():
    assert _parse_response("ALLOW - harmless package resolution check")[0] is True


def test_parse_response_accepts_leading_deny_verdict():
    allow, reason = _parse_response("DENY - deletes user files")

    assert allow is False
    assert reason == "deletes user files"


def test_parse_response_still_rejects_unclear_text():
    assert _parse_response("I cannot determine this from context.") == (
        False,
        "could not parse auditor response",
    )
