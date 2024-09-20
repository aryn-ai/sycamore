def test_configuration_sha():
    from hashlib import sha256
    import configuration

    with open(configuration.__file__, "rb") as f:
        bytes = f.read()
        sha = sha256(bytes).hexdigest()
        # If the change was intentional, update the hash
        # Think about whether you have to make the change since everyone with a custom config will need to update it
        assert sha == "1c12bf1d3437a494d24910bf2073b7461ac6dfa4c37c25d9929878fc35be9d40", f"hash mismatch got {sha}"
