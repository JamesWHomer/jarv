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
