from database.database import load_faces

class Retriever:
    def retrieve(self, id=False, name=True, image=False, embedding=True):
        results = load_faces()
        filtered_results = []

        for id_, name_, image_, embedding_ in results:
            item = {}
            if id:
                item['id'] = id_
            if name:
                item['name'] = name_
            if image:
                item['image'] = image_
            if embedding:
                item['embedding'] = embedding_
            filtered_results.append(item)

        return filtered_results
