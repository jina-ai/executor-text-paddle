__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional

import numpy as np
import paddlehub as hub
from jina import Executor, DocumentArray, requests


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
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        model_name: Optional[str] = 'ernie_tiny',
        on_gpu: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = hub.Module(name=model_name)
        self.on_gpu = on_gpu

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        """Encode doc content into vector representation.

        :param docs: `DocumentArray` passed from the previous ``Executor``.
        :param kwargs: Additional key value arguments.
        """
        for doc in docs:
            pooled_features = []
            results = self.model.get_embedding(
                np.atleast_2d(doc.content).reshape(-1, 1).tolist(), use_gpu=self.on_gpu
            )
            for emb in results:
                pooled_feature, seq_feature = emb
                pooled_features.append(pooled_feature)
            doc.embedding = np.asarray(pooled_features)
