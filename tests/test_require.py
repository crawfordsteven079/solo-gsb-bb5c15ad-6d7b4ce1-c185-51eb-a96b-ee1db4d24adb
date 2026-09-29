"""Tests for required-pointer coverage enforcement on POST /redact."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _post(payload):
    return client.post("/redact", json=payload)


# --------------------------------------------------------------------------- #
# Compatibility: no 'require' declared
# --------------------------------------------------------------------------- #


def test_no_require_field_behaviour_unchanged():
    resp = _post({"data": {"a": 1, "b": 2}, "rules": [{"pointer": "/a", "action": "mask"}]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["redacted"] == {"a": "***REDACTED***", "b": 2}
    assert set(body) == {"redacted", "audit", "policy"}


def test_require_empty_list_behaves_like_absent():
    resp = _post({"data": {"a": 1}, "rules": [], "require": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted"] == {"a": 1}


# --------------------------------------------------------------------------- #
# Coverage semantics
# --------------------------------------------------------------------------- #


def test_covered_by_action_on_the_node_itself():
    resp = _post(
        {
            "data": {"email": "a@x.io"},
            "rules": [{"pointer": "/email", "action": "mask"}],
            "require": ["/email"],
        }
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted"] == {"email": "***REDACTED***"}


def test_covered_by_action_on_an_ancestor():
    resp = _post(
        {
            "data": {"users": [{"email": "a@x.io"}]},
            "rules": [{"pointer": "/users", "action": "remove"}],
            "require": ["/users/0/email"],
        }
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted"] == {}


def test_covered_by_sha256_and_root_pointer():
    resp = _post(
        {
            "data": {"ssn": "123"},
            "rules": [{"pointer": "", "action": "sha256"}],
            "require": ["", "/ssn"],
        }
    )
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json()["redacted"], str)


def test_action_on_descendant_does_not_cover_parent():
    resp = _post(
        {
            "data": {"card": {"cvv": "1", "pan": "2"}},
            "rules": [{"pointer": "/card/cvv", "action": "mask"}],
            "require": ["/card"],
        }
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["uncovered"] == ["/card"]


def test_missing_paths_are_ignored():
    resp = _post(
        {
            "data": {"a": 1},
            "rules": [{"pointer": "/a", "action": "mask"}],
            "require": ["/a", "/nope", "/nope/deeper", "/a/b"],
        }
    )
    assert resp.status_code == 200, resp.text


def test_missing_array_tokens_are_ignored():
    # Out-of-range index, '-' (past-the-end), and leading-zero tokens never
    # resolve against an array -> ignored, not uncovered.
    resp = _post(
        {
            "data": {"items": [{"v": 1}]},
            "rules": [{"pointer": "/items/0/v", "action": "mask"}],
            "require": ["/items/0/v", "/items/9", "/items/-", "/items/01", "/items/x"],
        }
    )
    assert resp.status_code == 200, resp.text


def test_uncovered_fails_whole_request_without_data_or_audit():
    resp = _post(
        {
            "data": {"email": "a@x.io", "ssn": "123"},
            "rules": [{"pointer": "/email", "action": "mask"}],
            "require": ["/email", "/ssn"],
        }
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "required_pointers_uncovered"
    assert detail["uncovered"] == ["/ssn"]
    # No redacted document and no audit trail leak on failure.
    assert "redacted" not in resp.json()
    assert "audit" not in resp.json()


def test_uncovered_listed_in_declaration_order():
    resp = _post(
        {
            "data": {"a": 1, "b": 2, "c": 3},
            "rules": [{"pointer": "/b", "action": "mask"}],
            "require": ["/c", "/b", "/a", "/c"],
        }
    )
    assert resp.status_code == 422
    # Declaration order preserved; duplicates reported per declaration.
    assert resp.json()["detail"]["uncovered"] == ["/c", "/a", "/c"]


def test_array_deletion_keeps_original_indices_for_coverage():
    # /items/0 is removed; coverage of /items/2/v must still be judged
    # against the ORIGINAL third element.
    resp = _post(
        {
            "data": {"items": [{"v": "a"}, {"v": "b"}, {"v": "c"}]},
            "rules": [
                {"pointer": "/items/0", "action": "remove"},
                {"pointer": "/items/2/v", "action": "mask"},
            ],
            "require": ["/items/2/v", "/items/0/v"],
        }
    )
    # /items/2/v covered by its own mask; /items/0/v covered via ancestor
    # /items/0 being removed.
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted"]["items"] == [{"v": "b"}, {"v": "***REDACTED***"}]


def test_uncovered_after_array_deletion_uses_original_indices():
    resp = _post(
        {
            "data": {"items": [{"v": "a"}, {"v": "b"}]},
            "rules": [{"pointer": "/items/0", "action": "remove"}],
            "require": ["/items/1/v"],
        }
    )
    # Original /items/1/v was never acted upon -> uncovered, even though the
    # surviving element shifts to index 0 in the output.
    assert resp.status_code == 422
    assert resp.json()["detail"]["uncovered"] == ["/items/1/v"]


def test_conflict_resolution_still_applies_with_require():
    # Later-declared, equally specific rule wins; the winner is what counts
    # for coverage.
    resp = _post(
        {
            "data": {"a": 1},
            "rules": [
                {"pointer": "/a", "action": "mask"},
                {"pointer": "/a", "action": "sha256"},
            ],
            "require": ["/a"],
        }
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["audit"][0]["action"] == "sha256"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_invalid_required_pointer_returns_422():
    resp = _post({"data": {"a": 1}, "rules": [], "require": ["not-a-pointer"]})
    assert resp.status_code == 422


def test_wildcard_segments_rejected_in_require():
    for pointer in ("/users/*/email", "/**/ssn", "/users/**"):
        resp = _post({"data": {}, "rules": [], "require": [pointer]})
        assert resp.status_code == 422, pointer


def test_escaped_segments_allowed_in_require():
    # '~1' and '~0' are exact segments (RFC 6901 escaping), not wildcards.
    resp = _post(
        {
            "data": {"a/b": {"~key": 1}},
            "rules": [{"pointer": "/a~1b/~0key", "action": "mask"}],
            "require": ["/a~1b/~0key"],
        }
    )
    assert resp.status_code == 200, resp.text


def test_require_with_extra_field_still_422():
    resp = _post({"data": {}, "rules": [], "require": [], "bogus": 1})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Policy-backed requests
# --------------------------------------------------------------------------- #


def _put_policy(name, rules, etag=None):
    headers = {"Content-Type": "application/json"}
    if etag is not None:
        headers["If-Match"] = etag
    return client.put(f"/policies/{name}", json={"rules": rules}, headers=headers)


def test_require_with_policy_reference_latest_and_pinned():
    resp = _put_policy("req-pol", [{"pointer": "/email", "action": "mask"}])
    assert resp.status_code == 201
    etag = resp.headers["etag"]

    resp = _put_policy(
        "req-pol",
        [
            {"pointer": "/email", "action": "mask"},
            {"pointer": "/ssn", "action": "remove"},
        ],
        etag=etag,
    )
    assert resp.status_code == 200

    doc = {"email": "a@x.io", "ssn": "123"}

    # Latest version (v2) covers both declared pointers.
    resp = _post({"data": doc, "policy": {"name": "req-pol"}, "require": ["/email", "/ssn"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["policy"] == {"name": "req-pol", "version": 2}

    # Pinned v1 does not touch /ssn -> the request fails.
    resp = _post(
        {
            "data": doc,
            "policy": {"name": "req-pol", "version": 1},
            "require": ["/email", "/ssn"],
        }
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["uncovered"] == ["/ssn"]


def test_require_with_missing_policy_still_404():
    resp = _post(
        {"data": {"a": 1}, "policy": {"name": "nope"}, "require": ["/a"]}
    )
    assert resp.status_code == 404
