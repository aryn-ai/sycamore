def test_configuration_sha():
    from hashlib import sha256
    import configuration

    with open(configuration.__file__, "rb") as f:
        bytes = f.read()
        sha = sha256(bytes).hexdigest()
        # If the change was intentional, update the hash
        # Think about whether you have to make the change since everyone with a custom config will need to update it
        assert sha == "cf89894116604d4b002f2c5b6c9acf25982bf764310a9a50827608dcdc6b1b2c", f"hash mismatch got {sha}"
