from trackey.core.factories.store import StoreBuilder
builder = StoreBuilder("base_pipeline.yaml")
for name, store_cfg in builder.cfg.get("stores", {}).items():
    repo = store_cfg.get("repository")
    print("Repo config passed to validate:", repo)
