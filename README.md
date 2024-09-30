# LLM-GNN

## Getting started

> Data and raw text from https://github.com/agiresearch/InstructGLM and https://github.com/XiaoxinHe/TAPE

### Environment: 
> python==3.8  
  > torch==1.13.1  
  > tqdm==4.65.0  
  > transformers==4.28.0  
  > numpy==1.23.5  
 requests==2.28.1  
  > PyYAML==6.0   
    dgl  
    googledrivedownloader  
    category_encoders  
    einops   
    dgl
### Essential hyperparam:  

* > epoch  : If you use text_embeddings, you must construct a projection layer to make the dimensionality of the embeddings consistent with the model's dimensionality. Constrained by the projection layer, this may require multiple training epochs, or an increase in the lr

* > text_embedding or raw_text : text_embedding better performance

* > tau  : Controlling the distribution of gumbel_softmax 

* > clip :  LLM  only requires a clipping value of clip=1.0. However, in cases where both a de_model and a proj_model are present, it is recommended to combine them and increase the clipping value（5，10，20）
  


### workflow
> train_teather.py

> pretrain.py
