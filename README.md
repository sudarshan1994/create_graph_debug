# create_graph_debug
1) A minimal example using Bert embedding layer to show that create_graph argument to the `.backward()` function leads to different gradients <br/>
2) To run:`python test.py` <br/>

3)`models` folder contains the weights of pre-trained BertEmbeddinglayer and DecoderLayer. Note for randomly intialized BertEmbeddingLayer and DecoderLayer we do not observe this gradient discrepany<br/>
