def test_configuration_sha():
    from hashlib import sha256
    import configuration

    with open(configuration.__file__, "rb") as f:
        bytes = f.read()
        sha = sha256(bytes).hexdigest()
        # If the change was intentional, update the hash
        # Think about whether you have to make the change since everyone with a custom config will need to update it
        assert sha == "c8805d3e9a641a6e29a947e179a8eb7441567ecbbc238bc7a39c8ea2e2c811cf", f"hash mismatch got {sha}"
