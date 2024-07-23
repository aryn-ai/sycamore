import os
import pickle


class QueryDataInspector:

    def __init__(self, base_path: str) -> None:
        super().__init__()
        self._base_path = base_path

    def get_all_docs(self):
        result = {}
        directories = next(os.walk(self._base_path))[1]
        for directory in directories:
            result[directory] = self.get_documents_in_dir(self._base_path + directory)
        return result

    @staticmethod
    def get_documents_in_dir(path: str):
        files = os.listdir(path)
        documents = []
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, "rb") as f:
                document = pickle.load(f)
                documents.append(document)
        return documents

    def get_documents_for_node(self, node_id: str):
        path = self._base_path + "/" + node_id
        return self.get_documents_in_dir(path)

    def get_counts(self):
        result = {}
        for node, docs in self.get_all_docs().items():
            result[node] = len(docs)
        return result


## Sample usage
inspector = QueryDataInspector("/Users/vinayakthapliyal/tmp/intermediate/342762a8-84aa-4d44-afb2-44ebd445c209/")
print(inspector.get_counts())
