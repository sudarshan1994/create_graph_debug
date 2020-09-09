# create_graph_debug
A minimal example using Bert embedding layer to show that create_graph argument to the `.backward()` function leads to different gradients <br/>
To run:`python test.py` <br/>
`models` folder contains the weights of pre-trained BertEmbeddinglayer and DecoderLayer. Note for randomly intialized BertEmbeddingLayer and DecoderLayer we do not observe this gradient discrepany<br/>
