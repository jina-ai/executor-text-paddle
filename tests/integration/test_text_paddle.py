import pytest
from jina import Document, DocumentArray, Flow

from jinahub.encoder.text_paddle import TextPaddleEncoder


@pytest.fixture(scope='function')
def flow():
    return Flow().add(uses=TextPaddleEncoder)


@pytest.fixture(scope='function')
def content():
    return 'hello world'


@pytest.fixture(scope='function')
def document_array(content):
    return DocumentArray([Document(content=content)])


@pytest.fixture(scope='function')
def parameters(content):
    return {'traverse_paths': ['r'], 'batch_size': 10}


def test_text_paddle(flow, content, document_array, parameters):
    def validate(resp):
        for doc in resp.docs:
            assert doc.embedding.shape == (1024,)
            assert doc.embedding.all()

    with flow as f:
        f.index(inputs=document_array, on_done=validate)
