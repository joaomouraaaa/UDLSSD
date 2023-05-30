Code for the project: Unsupervised deep learning methods for SAR ship detection 

With the avaiable code it is possible to run the models for your own data.
Small amount of data is also provided.

Link to download trained files: (downloaded files should be added to "models" folder)
https://drive.google.com/drive/folders/1zkLqgbw2zxfE5Z3xpWf919LO98ikMEJY?usp=share_link

Link to download paired SAR-Ship-Dataset images with their supervised segmentation:
https://drive.google.com/drive/folders/1rWqjdGdb1Mj9LaJdit7bypM0A5XXopBu?usp=share_link



Set of instructions to run the models:


Run the cyclegan model:

  i) Start by adding your files to CycleGAN/data/..

  ii) Run: !python3 CycleGAN/code/main_cycle_gan.py

Run UDSEP model:

  i) Start by adding your files to UDSEP/data/images

  ii) Use the cyclegan model to generate images with DSEP

      !python3 CycleGAN/code/generate_predictions.py

      !python3 UDSEP/code/DSEP_transformation.py

  iii) Run !python3 UDSEP/code/main_unet.py



Test the results:

  i) Run !python3 data/test/test_results.py “model” “model_number”

  Examples:  

    !python3 data/test/test_results.py unet 150

    !python3 data/test/test_results.py cyclegan g_model_AtoB_016894.h5
