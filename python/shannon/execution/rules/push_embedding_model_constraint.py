from shannon.execution import (Node, Rule)


class PushEmbeddingModelConstraint(Rule):

    class Constraint:
        def __init__(self):
            self.max_seq_length = None

        def push(self, node: Node) -> None:
            from shannon.execution.transforms import (Embedding, Partition)
            match node:
                case Embedding():

                    from sentence_transformers import SentenceTransformer
                    transformer = SentenceTransformer(node.model_name)
                    self.max_seq_length = transformer.max_seq_length
                    dimension = transformer.get_sentence_embedding_dimension()
                    del transformer

                    # Load around 10MB data per batch
                    # TODO, we should have a more accurate way for computing
                    #  batch size based on dimension and GPU memory
                    if node.batch_size is None:
                        batch_size = int(10 * 1024 * 1024 / (8 * dimension))
                        node.set_batch_size(batch_size)
                case Partition() if node.max_partition is None:
                    node.set_max_partition(self.max_seq_length)
                case _:
                    pass

    def __call__(self, plan: Node) -> Node:
        constraint = PushEmbeddingModelConstraint.Constraint()
        plan.traverse_down(constraint.push)
        return plan
