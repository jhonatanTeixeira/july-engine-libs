class BridgeInterface:

    async def start(self):
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError

    async def process_openai_chat(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_anthropic_message(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_tts(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_stt(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_image_generation(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_image_edit(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_image_resize(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_image_remove_background(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_image_description(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_video_description(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_face_extraction(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_face_sync_batch(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_embeddings(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_web_search(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_code_search(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_search_web(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_search_code(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_add(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_batch_add(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_search(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_vector_add(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_update(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_delete(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_list(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_rag_smart_search(self, payload: dict, headers: dict):
        raise NotImplementedError

    async def process_search_and_scrape(self, results: list, query: str, headers: dict, describe_model: str = None):
        raise NotImplementedError

    async def process_resource_check(self, payload: dict):
        raise NotImplementedError

    async def process_pdf_extract(self, pdf_bytes: bytes):
        raise NotImplementedError

    async def process_video_generation(self, payload: dict, headers: dict):
        raise NotImplementedError
