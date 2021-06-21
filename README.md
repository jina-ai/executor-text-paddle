# TextPaddleEncoder

**TextPaddleEncoder** is a class that wraps the text embedding functionality from the **PaddlePaddle** and **PaddleHub**.


`TextPaddleEncoder` encode text stored in the `text` attribute of the [**Document**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) and saves the encoding in the embedding attribute.



## Prerequisites

None


## Usages

### Via JinaHub (🚧W.I.P.)

Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
        uses='jinahub+docker://TextPaddleEncoder',
        volumes='/your_home_folder/.cache/paddle:/root/.cache/paddle')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TextPaddleEncoder'
    volumes: '/your_home_folder/.cache/paddle:/root/.cache/paddle'
```


### Via Pypi

1. Install the `TextPaddleEncoder`

	```bash
	pip install git+https://github.com/jina-ai/executor-text-paddle.git
	```

1. Use `TextPaddleEncoder` in your code

	```python
	from jinahub.encoder.text_paddle import TextPaddleEncoder
	from jina import Flow
	
	f = Flow().add(uses=TextPaddleEncoder)
	```


### Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-text-paddle.git
	cd executor-text-paddle
	docker build -t jinahub-text-paddle .
	```

1. Use `jinahub-text-paddle` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(
	        uses='docker://jinahub-text-paddle:latest',
	        volumes='/your_home_folder/.cache/paddle:/root/.cache/paddle')
	```
	


## Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TextPaddleEncoder',
               volumes='/Users/nanwang/.cache/paddle:/root/.cache/paddle')


def check_resp(resp):
    for _doc in resp.data.docs:
        doc = Document(_doc)
        print(f'embedding shape: {doc.embedding.shape}')


with f:
    f.post(on='foo',
           inputs=Document(text='hello world'),
           on_done=check_resp)
	    
```


### Inputs 

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `text`.

### Returns

[Documents](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) with `embedding` fields filled with an `ndarray` of the shape `1024` (depends on the model) with `dtype=nfloat32`.



## Reference
- https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
