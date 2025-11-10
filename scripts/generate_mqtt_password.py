#!/usr/bin/env python3
"""
Generate Mosquitto-compatible password hashes without requiring mosquitto_passwd.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
from pathlib import Path

ITERATIONS = 101  # matches mosquitto_passwd default for PBKDF2
SALT_BYTES = 12


def build_hash(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt or os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), salt_bytes, ITERATIONS)
    return f"$7${ITERATIONS}${base64.b64encode(salt_bytes).decode()}${base64.b64encode(digest).decode()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update mosquitto password files.")
    parser.add_argument("--username", default="hcai_operator")
    parser.add_argument("--password")
    parser.add_argument("--output", type=Path, default=Path("mosquitto/passwordfile"))
    args = parser.parse_args()

    password = args.password or getpass.getpass(prompt=f"MQTT password for {args.username}: ")
    hashed = build_hash(password)
    line = f"{args.username}:{hashed}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(line, encoding="utf-8")
    print(f"Wrote credentials for {args.username} to {args.output}")


if __name__ == "__main__":
    main()
