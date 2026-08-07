"""Run one command with a user-bound DPAPI secret in its environment.

The encrypted blob and environment-variable name are command-line metadata;
the decrypted value is never placed in argv, printed, or written to disk.
This helper is Windows-only and is intended to be invoked from Git Bash.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
from ctypes import POINTER, Structure, byref, c_char, c_void_p, cast, string_at
from ctypes.wintypes import DWORD
from pathlib import Path

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _DataBlob(Structure):
    _fields_ = [("cbData", DWORD), ("pbData", POINTER(c_char))]


def decrypt_dpapi_blob(path: Path) -> str:
    """Decrypt a raw CryptProtectData blob for the current Windows user."""

    if os.name != "nt":
        raise RuntimeError("DPAPI secret execution is supported only on Windows")
    encrypted = path.resolve(strict=True).read_bytes()
    if not encrypted:
        raise ValueError("DPAPI secret file is empty")
    buffer = (c_char * len(encrypted)).from_buffer_copy(encrypted)
    input_blob = _DataBlob(len(encrypted), cast(buffer, POINTER(c_char)))
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    success = crypt32.CryptUnprotectData(
        byref(input_blob),
        None,
        None,
        None,
        None,
        0,
        byref(output_blob),
    )
    if not success:
        raise OSError("DPAPI decryption failed for the current Windows user")
    try:
        value = string_at(output_blob.pbData, output_blob.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(cast(output_blob.pbData, c_void_p))
    value = value.strip()
    if not value:
        raise ValueError("decrypted DPAPI secret is empty")
    if "\x00" in value:
        raise ValueError("decrypted DPAPI secret contains a NUL character")
    return value


def run_with_secret(
    *,
    secret_file: Path,
    env_names: list[str],
    command: list[str],
) -> int:
    """Run ``command`` while keeping the decrypted secret out of argv."""

    if not env_names or any(not _ENV_NAME.fullmatch(name) for name in env_names):
        raise ValueError("env-name must be a valid environment-variable name")
    if len(set(env_names)) != len(env_names):
        raise ValueError("env-name values must be unique")
    if not command or any(not part for part in command):
        raise ValueError("a non-empty command is required")
    child_env = os.environ.copy()
    secret = decrypt_dpapi_blob(secret_file)
    for env_name in env_names:
        child_env[env_name] = secret
    completed = subprocess.run(command, env=child_env, check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-file", required=True, type=Path)
    parser.add_argument("--env-name", required=True, action="append")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return run_with_secret(
        secret_file=args.secret_file,
        env_names=list(args.env_name),
        command=command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
