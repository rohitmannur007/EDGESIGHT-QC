from edgesight.persistence import EdgeSightStore


def test_versioned_store_round_trip(tmp_path):
    store = EdgeSightStore(tmp_path / "operations.db", allow_sqlite_fallback=True)
    batch_id = store.create_batch(1, {"line": "A"})
    result_id = store.save_result(
        status="complete", prediction={"decision": "accept"}, batch_id=batch_id
    )

    assert store.get_result(result_id)["prediction"] == {"decision": "accept"}
    review_id = store.add_review(result_id, "operator", "accepted", "looks good")
    assert store.list_reviews(result_id)[0]["id"] == review_id
    assert store.query("SELECT MAX(version) AS version FROM schema_version")[0]["version"] == 3
