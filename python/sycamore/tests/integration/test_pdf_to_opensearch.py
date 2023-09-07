from typing import List

import sycamore
from data import Element
from sycamore.execution.transforms import PdfPartitionerOptions
from sycamore.execution.transforms.llms.llms import OpenAIModels, OpenAI
from sycamore.tests.config import TEST_DIR


def test_pdf_to_opensearch():
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    index_settings = {
        "body": {
            "settings": {
                "index.knn": True,
                "number_of_shards": 5,
                "number_of_replicas": 1,
            },
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "nmslib"},
                    },
                    "text": {"type": "text"},
                }
            },
        }
    }

    title_context_template = """
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

    author_context_template = """
        ELEMENT 1: Jupiter's Moons
        ELEMENT 2: Ganymede 2020
        ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
        ELEMENT 4: From Wikipedia, the free encyclopedia
        ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
        =========
        Audi Laupe, Serena K. Goldberg

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
        Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran

        """

    abstract_context_template = """    
        ELEMENT 1: Attention Is All You Need
        ELEMENT 2: Ashish Vaswani∗ Google Brain avaswani@google.com
        ELEMENT 3: Noam Shazeer∗ Google Brain noam@google.com
        ELEMENT 4: Niki Parmar∗ Google Research nikip@google.com
        ELEMENT 5: Jakob Uszkoreit∗ Google Research usz@google.com
        ELEMENT 6: Llion Jones∗ Google Research llion@google.com
        ELEMENT 7: Aidan N. Gomez∗ † University of Toronto aidan@cs.toronto.edu
        ELEMENT 8: Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com
        ELEMENT 9: Abstract
        ELEMENT 10: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
        ========
        "abstract": The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
        ELEMENT 1: Ray: A Distributed Framework for Emerging AI Applications
        ELEMENT 2: Philipp Moritz∗, Robert Nishihara∗, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, Ion Stoica University of California, Berkeley
        ELEMENT 3: 8 1 0 2
        ELEMENT 4: Abstract
        ELEMENT 5: and their use in prediction. These frameworks often lever- age specialized hardware (e.g., GPUs and TPUs), with the goal of reducing training time in a batch setting. Examples include TensorFlow [7], MXNet [18], and PyTorch [46]. The promise of AI is, however, far broader than classi- cal supervised learning. Emerging AI applications must increasingly operate in dynamic environments, react to changes in the environment, and take sequences of ac- tions to accomplish long-term goals [8, 43]. They must aim not only to exploit the data gathered, but also to ex- plore the space of possible actions. These broader require- ments are naturally framed within the paradigm of rein- forcement learning (RL). RL deals with learning to oper- ate continuously within an uncertain environment based on delayed and limited feedback [56]. RL-based systems have already yielded remarkable results, such as Google's AlphaGo beating a human world champion [54], and are beginning to ﬁnd their way into dialogue systems, UAVs [42], and robotic manipulation [25, 60].
        ELEMENT 6: p e S 0 3
        ELEMENT 7: ]
        ELEMENT 8: C D . s c [
        ELEMENT 9: 2 v 9 8 8 5 0 . 2 1 7 1 : v i X r a
        ELEMENT 10: The next generation of AI applications will continuously interact with the environment and learn from these inter- actions. These applications impose new and demanding systems requirements, both in terms of performance and ﬂexibility. In this paper, we consider these requirements and present Ray—a distributed system to address them. Ray implements a uniﬁed interface that can express both task-parallel and actor-based computations, supported by a single dynamic execution engine. To meet the perfor- mance requirements, Ray employs a distributed scheduler and a distributed and fault-tolerant store to manage the system's control state. In our experiments, we demon- strate scaling beyond 1.8 million tasks per second and better performance than existing specialized systems for several challenging reinforcement learning applications.
        ========
        "abstract": The next generation of AI applications will continuously interact with the environment and learn from these inter- actions. These applications impose new and demanding systems requirements, both in terms of performance and ﬂexibility. In this paper, we consider these requirements and present Ray—a distributed system to address them. Ray implements a uniﬁed interface that can express both task-parallel and actor-based computations, supported by a single dynamic execution engine. To meet the perfor- mance requirements, Ray employs a distributed scheduler and a distributed and fault-tolerant store to manage the system's control state. In our experiments, we demon- strate scaling beyond 1.8 million tasks per second and better performance than existing specialized systems for several challenging reinforcement learning applications.
        """

    def prompt_formatter(elements: List[Element]) -> str:
        query = ""
        for i in range(len(elements)):
            query += f"ELEMENT {i + 1}: {elements[i].get('content').get('text')}\n"
        return query

    paths = str(TEST_DIR / "resources/data/sort_benchmark/")

    openai_llm = OpenAI(OpenAIModels.TEXT_DAVINCI.value,
                        "api-key")

    context = sycamore.init()
    ds = (
        context.read.binary(paths, binary_format="pdf")
        .partition(max_partition=256, options=PdfPartitionerOptions())
        .llm_extract_entity(
            entity_to_extract="title",
            llm=openai_llm,
            prompt_template=title_context_template,
            prompt_formatter=prompt_formatter,
        )
        .llm_extract_entity(
            entity_to_extract="authors",
            llm=openai_llm,
            prompt_template=author_context_template,
            prompt_formatter=prompt_formatter,
            model_name=OpenAIModels.TEXT_DAVINCI.value,
        ).llm_extract_entity(
            entity_to_extract="abstract",
            llm=openai_llm,
            prompt_template=abstract_context_template,
            prompt_formatter=prompt_formatter,
            model_name=OpenAIModels.TEXT_DAVINCI.value,
        ))

    ds.show()
