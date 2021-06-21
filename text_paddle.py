__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Any, Dict, Union

import numpy as np
import paddlehub as hub
from jina import Executor, DocumentArray, requests


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class TextPaddleEncoder(Executor):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Internally, :class:`TextPaddlehubEncoder` wraps the Ernie module from paddlehub.
    https://github.com/PaddlePaddle/PaddleHub
    For models' details refer to
        https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel

    :param model_name: the name of the model. Supported models include
        ``ernie``, ``ernie_tiny``, ``ernie_v2_eng_base``, ``ernie_v2_eng_large``,
        ``bert_chinese_L-12_H-768_A-12``, ``bert_multi_cased_L-12_H-768_A-12``,
        ``bert_multi_uncased_L-12_H-768_A-12``, ``bert_uncased_L-12_H-768_A-12``,
        ``bert_uncased_L-24_H-1024_A-16``, ``chinese-bert-wwm``,
        ``chinese-bert-wwm-ext``, ``chinese-electra-base``,
        ``chinese-electra-small``, ``chinese-roberta-wwm-ext``,
        ``chinese-roberta-wwm-ext-large``, ``rbt3``, ``rbtl3``
    :param on_gpu: If use gpu to get the output.
    :param default_batch_size: fallback batch size in case there is not batch size sent in the request
    :param default_traversal_path: fallback traversal path in case there is not traversal path sent in the request
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        model_name: Optional[str] = 'ernie_tiny',
        on_gpu: bool = False,
        default_batch_size: int = 32,
        default_traversal_paths: Union[str, List[str]] = 'r',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.on_gpu = on_gpu
        self.model = hub.Module(name=model_name)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths

    def _get_input_data(self, docs, parameters):
        trav_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = []
        for trav_path in trav_paths:
            flat_docs.extend(docs.traverse_flat(trav_path))

        # filter out documents without text
        filtered_docs = [doc for doc in flat_docs if doc.text is not None]

        return _batch_generator(filtered_docs, batch_size)

    @requests
    def encode(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Encode doc content into vector representation.

        :param docs: `DocumentArray` passed from the previous ``Executor``.
        :param parameters: dictionary to define the `traversal_path` and the `batch_size`. For example,
            `parameters={'traversal_path': 'r', 'batch_size': 10}` will override the `self.default_traversal_path` and
            `self.default_batch_size`.
        :param kwargs: Additional key value arguments.
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            for batch_of_docs in document_batches_generator:
                pooled_features = []
                contents = [[doc.content] for doc in batch_of_docs]
                print(f"\n\n{contents}\n\n")
                results = self.model.get_embedding(contents, use_gpu=self.on_gpu)
                for emb in results:
                    pooled_feature, seq_feature = emb
                    pooled_features.append(pooled_feature)
                for doc, feature in zip(batch_of_docs, pooled_features):
                    doc.embedding = np.asarray(feature)
