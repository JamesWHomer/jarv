from jarv.tool_outputs import (
    TOOL_FAILURE_PREFIXES,
    flatten_content_text,
    image_data_url,
    tool_output_failed,
)


def test_tool_output_failed_detects_every_failure_prefix():
    samples = {
        "[error:": "[error: something broke]",
        "[tool argument error:": "[tool argument error: invalid JSON]",
        "[unknown tool:": "[unknown tool: frobnicate]",
        "[edit error:": "[edit error: old_text not found]",
        "[edit denied": "[edit denied by user]",
        "[read error:": "[read error: no such file]",
        "[read image unavailable:": "[read image unavailable: no image capability]",
        "[web error:": "[web error: no search results found]",
        "[tool disabled:": "[tool disabled: web_search]",
    }
    assert set(samples) == set(TOOL_FAILURE_PREFIXES)
    for output in samples.values():
        assert tool_output_failed(output), output


def test_tool_output_failed_detects_cancellation_anywhere():
    assert tool_output_failed("command run cancelled by user")


def test_tool_output_failed_ignores_successful_outputs():
    for output in (
        "",
        "working tree clean",
        "[READ RESULT]\nReturned size: 5",
        "Query: python\n\n1. Result",
    ):
        assert not tool_output_failed(output), output


def test_flatten_content_text_handles_history_blocks():
    content = [
        {"type": "text", "text": "hello"},
        {"content": "nested content"},
        {"type": "reasoning"},
        "bare string",
    ]

    assert flatten_content_text(content) == (
        "hello\nnested content\n[reasoning]\nbare string"
    )


def test_flatten_content_text_handles_tool_output_blocks():
    content = [
        {"type": "input_text", "text": "[READ RESULT]"},
        {"type": "input_image", "image_url": image_data_url("image/png", b"123456")},
    ]

    assert flatten_content_text(content) == (
        "[READ RESULT]\n[image output 1: image/png, 6 bytes]"
    )


def test_flatten_content_text_passes_strings_through():
    assert flatten_content_text("plain") == "plain"
    assert flatten_content_text(None) == ""
