def test_configuration_sha():
    from hashlib import sha256
    import configuration

    with open(configuration.__file__, "rb") as f:
        bytes = f.read()
        sha = sha256(bytes).hexdigest()
        # If the change was intentional, update the hash
        # Think about whether you have to make the change since everyone with a custom config will need to update it
        assert sha == "0d5a24a1edfb9e523814dc847fa34cdcde2d2e78aff03e3f4489755e12be2c54", f"hash mismatch got {sha}"
