from stage3_agent.memory import MemoryStore


def test_memory_namespace_isolation_and_invalidation(tmp_path):
    path = tmp_path / "memory.sqlite3"
    left = MemoryStore(path, "run:full:user-left")
    right = MemoryStore(path, "run:full:user-right")
    item = left.write({"key": "accessibility", "value": "wheelchair_accessible"}, 1, "setup")
    assert left.search("accessibility")
    assert right.search("accessibility") == []
    invalid = left.invalidate("accessibility", 2, "superseded")
    assert invalid[0]["memory_id"] == item["memory_id"]
    assert invalid[0]["status"] == "invalid"
    assert left.search("accessibility") == []
    left.close()
    right.close()
