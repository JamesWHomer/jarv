"""Standalone updates only install a regular executable for the current platform."""

import io
import tarfile

import pytest

from jarv.standalone import extract_executable, normalize_architecture


@pytest.mark.parametrize("spelling", ["arm64", "aarch64"])
@pytest.mark.parametrize("platform,expected", [
    ("linux", "aarch64"), ("macos", "arm64"), ("windows", "arm64"),
])
def test_arm_aliases_select_the_platform_asset(spelling, platform, expected):
    assert normalize_architecture(spelling, target_platform=platform) == expected


def test_extract_tar_copies_executable_bytes(tmp_path):
    archive_path = tmp_path / "jarv.tar.gz"
    payload = b"#!/bin/sh\necho jarv\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("jarv")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    executable = extract_executable(archive_path, tmp_path / "out", windows=False)
    assert executable.read_bytes() == payload


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_extract_tar_rejects_nonregular_executable(tmp_path, kind):
    archive_path = tmp_path / "jarv.tar.gz"
    sentinel = tmp_path / "keep.txt"
    sentinel.write_bytes(b"unchanged")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("jarv")
        member.type = kind
        member.linkname = "../keep.txt"
        archive.addfile(member)

    with pytest.raises(ValueError, match="regular file"):
        extract_executable(archive_path, tmp_path / "out", windows=False)

    assert sentinel.read_bytes() == b"unchanged"
    assert not (tmp_path / "out" / "jarv").exists()
