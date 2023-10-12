# ruff: noqa: E501
title_template = """
ELEMENT 1: Jupiter's Moons
ELEMENT 2: Ganymede 2020
ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
ELEMENT 4: From Wikipedia, the free encyclopedia
ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
=========
"Ganymede 2020"

ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
ELEMENT 2: Tarun Kalluri * UCSD
ELEMENT 3: Deepak Pathak CMU
ELEMENT 4: Manmohan Chandraker UCSD
ELEMENT 5: Du Tran Facebook AI
ELEMENT 6: https://tarun005.github.io/FLAVR/
ELEMENT 7: 2 2 0 2
ELEMENT 8: b e F 4 2
ELEMENT 9: ]
ELEMENT 10: V C . s c [
========
"FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"

"""

osrch_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}

idx_settings = {
    "body": {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "nmslib"},
                },
                "text_representation": {"type": "text"},
            }
        },
    }
}
