"""Tests for required-protection pointers (the request-level ``protect`` list).

A declared pointer that exists on the ORIGINAL document must be covered by an
actually executed mask/remove/sha256 action on itself or an ancestor;
otherwise the whole redaction fails. Non-existent pointers are ignored and
malformed pointers are a parameter error.
"""

from __future__ import annotations

import pytest

from app.redaction import (
    CoverageError,
    InvalidPointer,
    compile_rule,
    redact,
    uncovered_required,
)


def rule(index, pointer, action, name=None):
    return compile_rule(index=index, pointer=pointer, action=action, name=name)


# --------------------------------------------------------------------------- #
# Engine: coverage semantics
# --------------------------------------------------------------------------- #


def test_required_node_directly_covered():
    data = {"email": "a@x.io", "name": "n"}
    out, audit = redact(
        data, [rule(0, "/email", "mask")], required=["/email"]
    )
    assert out == {"email": "***REDACTED***", "name": "n"}
    assert audit[0]["pointer"] == "/email"


def test_required_node_covered_by_ancestor_subtree_action():
    data = {"card": {"number": "123", "cvv": "9"}}
    # No rule mentions either leaf; masking the ancestor covers both.
    out, audit = redact(
        data,
        [rule(0, "/card", "mask")],
        required=["/card/number", "/card/cvv"],
    )
    assert out == {"card": "***REDACTED***"}
    assert [e["pointer"] for e in audit] == ["/card"]


def test_descendant_action_does_not_cover_ancestor():
    data = {"card": {"number": "123"}}
    with pytest.raises(CoverageError) as exc:
        redact(
            data,
            [rule(0, "/card/number", "mask")],
            required=["/card"],
        )
    assert exc.value.uncovered == ["/card"]


def test_sha256_counts_as_coverage():
    data = {"ssn": "1"}
    out, _ = redact(data, [rule(0, "/ssn", "sha256")], required=["/ssn"])
    assert len(out["ssn"]) == 64


def test_root_pointer_covered_by_root_action():
    data = {"a": 1}
    out, audit = redact(data, [rule(0, "", "mask")], required=[""])
    assert out == "***REDACTED***"
    assert audit[0]["pointer"] == ""


def test_root_pointer_uncovered_without_action():
    with pytest.raises(CoverageError) as exc:
        # Root exists but nothing acts at all -> not covered.
        redact({"a": 1}, [], required=[""])
    assert exc.value.uncovered == [""]


def test_nonexistent_pointers_are_ignored():
    data = {"a": 1}
    out, audit = redact(
        data,
        [rule(0, "/a", "mask")],
        required=["/missing", "/missing/deep/1/x"],
    )
    assert out == {"a": "***REDACTED***"}
    assert [e["pointer"] for e in audit] == ["/a"]


def test_nonexistent_array_indices_are_ignored():
    data = {"items": [1, 2]}
    # Out of range, the RFC 6901 "-" append marker and leading-zero indices
    # do not resolve and must be ignored rather than covered.
    out, _ = redact(
        data,
        [rule(0, "/items/0", "mask")],
        required=["/items/2", "/items/-", "/items/01"],
    )
    assert out == {"items": ["***REDACTED***", 2]}


def test_uncovered_returns_declaration_order():
    data = {"z": 1, "q": 2, "a": 3}
    with pytest.raises(CoverageError) as exc:
        redact(
            data,
            [rule(0, "/a", "mask")],
            required=["/missing", "/z", "/a", "/q"],
        )
    # Missing ignored; /a covered; order follows the declaration.
    assert exc.value.uncovered == ["/z", "/q"]


def test_duplicate_pointers_reported_once():
    data = {"z": 1}
    with pytest.raises(CoverageError) as exc:
        redact(data, [], required=["/z", "/z", "/z"])
    assert exc.value.uncovered == ["/z"]


def test_glob_rule_can_cover_exact_required_pointer():
    data = {"users": [{"email": "a@x.io"}]}
    out, _ = redact(
        data,
        [rule(0, "/users/*/email", "mask")],
        required=["/users/0/email"],
    )
    assert out["users"][0]["email"] == "***REDACTED***"


def test_required_pointer_uses_rfc6901_escaping():
    data = {"a/b": {"c~d": "v"}}
    out, _ = redact(
        data,
        [rule(0, "/a~1b/c~0d", "remove")],
        required=["/a~1b/c~0d"],
    )
    assert out == {"a/b": {}}


def test_array_deletion_must_not_shift_required_index():
    # Original indices: /items/0 = a, /items/1 = b, /items/2 = c.
    # Removing index 0 must NOT make the surviving element (original 2) look
    # like "index 1 after the gap": /items/2 is judged on the original list.
    data = {"items": ["a", "b", "c"]}
    with pytest.raises(CoverageError) as exc:
        redact(
            data,
            [rule(0, "/items/0", "remove"), rule(1, "/items/1", "mask")],
            required=["/items/2"],
        )
    assert exc.value.uncovered == ["/items/2"]

    # Covering original index 2 explicitly satisfies the requirement, even
    # though the result array renumbers it to 1.
    out, _ = redact(
        data,
        [
            rule(0, "/items/0", "remove"),
            rule(1, "/items/1", "mask"),
            rule(2, "/items/2", "mask"),
        ],
        required=["/items/0", "/items/1", "/items/2"],
    )
    assert out == {"items": ["***REDACTED***", "***REDACTED***"]}


def test_coverage_failure_leaves_input_unmutated_and_no_result():
    data = {"k": "v", "keep": [1, 2, 3]}
    snapshot = {"k": "v", "keep": [1, 2, 3]}
    with pytest.raises(CoverageError):
        redact(
            data,
            [rule(0, "/k", "remove")],
            required=["/keep/1"],
        )
    assert data == snapshot


def test_uncovered_required_helper_directly():
    data = {"a": {"b": 1}, "x": 2}
    from app.redaction import build_hits

    hits = build_hits(data, [rule(0, "/a", "mask")])
    assert uncovered_required(data, ["/a/b", "/x", "/nope"], hits) == ["/x"]


# --------------------------------------------------------------------------- #
# Engine: malformed required pointers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pointer", ["no-leading-slash", "/users/*/email", "/**/x", "/a/*"])
def test_invalid_required_pointer_raises(pointer):
    data = {"a": {"x": 1}}
    with pytest.raises(InvalidPointer):
        redact(data, [rule(0, "/a/**", "mask")], required=[pointer])


def test_empty_required_list_is_compatible():
    data = {"a": 1}
    out, audit = redact(data, [rule(0, "/a", "mask")], required=[])
    assert out == {"a": "***REDACTED***"}
    assert audit


# --------------------------------------------------------------------------- #
# HTTP layer (POST /redact)
# --------------------------------------------------------------------------- #

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_http_protect_success_inline():
    resp = client.post(
        "/redact",
        json={
            "data": {"email": "a@x.io", "name": "n"},
            "rules": [{"pointer": "/email", "action": "mask"}],
            "protect": ["/email"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["redacted"] == {"email": "***REDACTED***", "name": "n"}
    assert [e["pointer"] for e in body["audit"]] == ["/email"]


def test_http_protect_ancestor_coverage():
    resp = client.post(
        "/redact",
        json={
            "data": {"card": {"number": "1", "cvv": "2"}},
            "rules": [{"pointer": "/card", "action": "remove"}],
            "protect": ["/card/number", "/card/cvv"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted"] == {}


def test_http_uncovered_returns_412_without_data_or_audit():
    resp = client.post(
        "/redact",
        json={
            "data": {"email": "a@x.io", "password": "p"},
            "rules": [{"pointer": "/email", "action": "mask"}],
            "protect": ["/email", "/password"],
        },
    )
    assert resp.status_code == 412
    detail = resp.json()["detail"]
    assert detail["error"] == "protect_paths_uncovered"
    assert detail["uncovered"] == ["/password"]
    # No redacted data and no audit leak on failure.
    assert "redacted" not in detail
    assert "audit" not in detail


def test_http_uncovered_order_and_ignore_missing():
    resp = client.post(
        "/redact",
        json={
            "data": {"z": 1, "a": 2},
            "rules": [{"pointer": "/a", "action": "mask"}],
            "protect": ["/missing", "/z", "/a", "/missing/2"],
        },
    )
    assert resp.status_code == 412
    assert resp.json()["detail"]["uncovered"] == ["/z"]


def test_http_protect_invalid_pointer_returns_422():
    # Not an RFC 6901 pointer.
    resp = client.post(
        "/redact",
        json={
            "data": {"a": 1},
            "rules": [{"pointer": "/a", "action": "mask"}],
            "protect": ["no-leading-slash"],
        },
    )
    assert resp.status_code == 422
    assert "protect" in resp.text


@pytest.mark.parametrize("pointer", ["/users/*/email", "/**/ssn", "/a/*"])
def test_http_protect_wildcard_segment_returns_422(pointer):
    resp = client.post(
        "/redact",
        json={
            "data": {"users": [{"email": "e"}]},
            "rules": [{"pointer": "/users/*/email", "action": "mask"}],
            "protect": [pointer],
        },
    )
    assert resp.status_code == 422, resp.text


def test_http_protect_entries_must_be_strings():
    resp = client.post(
        "/redact",
        json={"data": {"a": 1}, "rules": [], "protect": ["/a", 7]},
    )
    assert resp.status_code == 422


def test_http_protect_empty_list_is_compatible():
    resp = client.post(
        "/redact",
        json={"data": {"a": 1}, "rules": [{"pointer": "/a", "action": "mask"}], "protect": []},
    )
    assert resp.status_code == 200


def test_http_protect_omitted_keeps_legacy_response():
    resp = client.post(
        "/redact",
        json={"data": {"a": 1}, "rules": [{"pointer": "/a", "action": "mask"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"redacted", "audit", "policy"}


def test_http_protect_works_with_policy_and_pinned_version():
    created = client.put(
        "/policies/pii",
        json={"rules": [{"pointer": "/users/*/email", "action": "mask"}]},
    )
    assert created.status_code == 201
    etag = created.json()["etag"]
    updated = client.put(
        "/policies/pii",
        json={
            "rules": [
                {"pointer": "/users/*/email", "action": "mask"},
                {"pointer": "/users/*/password", "action": "remove"},
            ]
        },
        headers={"If-Match": etag},
    )
    assert updated.status_code == 200

    data = {"users": [{"email": "e", "password": "p"}]}

    # Latest version (v2): both fields are handled -> both requirements met.
    ok = client.post(
        "/redact",
        json={
            "data": data,
            "policy": {"name": "pii"},
            "protect": ["/users/0/email", "/users/0/password"],
        },
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["policy"] == {"name": "pii", "version": 2}

    # Pinned historical version (v1) does not remove password -> failure,
    # naming the concrete unresolved path; conflict arbitration is untouched.
    old = client.post(
        "/redact",
        json={
            "data": data,
            "policy": {"name": "pii", "version": 1},
            "protect": ["/users/0/email", "/users/0/password"],
        },
    )
    assert old.status_code == 412
    assert old.json()["detail"]["uncovered"] == ["/users/0/password"]


def test_http_protect_against_nonexistent_policy_still_404():
    resp = client.post(
        "/redact",
        json={
            "data": {"a": 1},
            "policy": {"name": "nope"},
            "protect": ["/a"],
        },
    )
    assert resp.status_code == 404


def test_http_protect_array_original_index_not_post_delete():
    resp = client.post(
        "/redact",
        json={
            "data": {"items": ["a", "b", "c"]},
            "rules": [
                {"pointer": "/items/0", "action": "remove"},
                {"pointer": "/items/1", "action": "mask"},
            ],
            "protect": ["/items/2"],
        },
    )
    assert resp.status_code == 412
    assert resp.json()["detail"]["uncovered"] == ["/items/2"]


def test_http_protect_extra_field_still_422():
    resp = client.post(
        "/redact",
        json={
            "data": {"a": 1},
            "rules": [],
            "protect": [],
            "bogus": 1,
        },
    )
    assert resp.status_code == 422
