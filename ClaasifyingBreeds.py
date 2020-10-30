#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *


# In[2]:


path = untar_data(URLs.PETS)


# In[4]:


files = get_image_files(path/"images")
len(files)


# In[5]:


pat = r'^(.*)_\d+.jpg'


# In[6]:


dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224),model_dir="/tmp/model/")


# In[7]:


dls.show_batch()


# In[8]:


learn = cnn_learner(dls, resnet34, metrics=error_rate,model_dir="/tmp/model/")


# In[9]:


learn.fine_tune(4, 3e-3)


# In[10]:


learn.show_results()


# In[11]:


learn.export(Path('/notebooks/export.pkl'))


# In[12]:


path = Path()
path.ls(file_exts='.pkl')


# In[13]:


learn_inf = load_learner(Path('/notebooks/export.pkl'))


# In[14]:


learn_inf.predict(Path('/storage/data/oxford-iiit-pet/images/american_bulldog_146.jpg'))


# In[15]:


learn_inf.dls.vocab


# In[16]:


#hide_output
from ipywidgets import widgets
btn_upload = widgets.FileUpload()
btn_upload


# In[17]:


img = PILImage.create(btn_upload.data[-1])


# In[19]:


out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[20]:


pred,pred_idx,probs = learn_inf.predict(img)


# In[21]:


lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[22]:


btn_run = widgets.Button(description='Classify')
btn_run


# In[23]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[24]:


btn_upload = widgets.FileUpload()


# In[25]:


from ipywidgets import *
VBox([widgets.Label('Select your bear!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:conda install -c conda-forge voila




