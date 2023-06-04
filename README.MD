Code for the project: Unsupervised deep learning methods for SAR ship detection

With this code, it is possible to run the models for your own data.  We also provide some of the data used in our experiments.

Link to download the model weights files: (these files should be added to "models" folder) https://drive.google.com/drive/folders/1zkLqgbw2zxfE5Z3xpWf919LO98ikMEJY?usp=share_link

Link to download SAR-Ship-Dataset segmentation data (images and their ground truth segmentation): https://drive.google.com/drive/folders/1rWqjdGdb1Mj9LaJdit7bypM0A5XXopBu?usp=share_link

Set of instructions to run the models: (Can be found on "notebook.ipynb") 

Train the cyclegan model:

i) Start by adding your files to CycleGAN/data/..

ii) Run: 

    !python3 CycleGAN/code/main_cycle_gan.py

Train the UDSEP model:

i) Start by adding your files to UDSEP/data/images

ii) Use the cyclegan model to generate images with DSEP

    !python3 CycleGAN/code/generate_predictions.py
    !python3 UDSEP/code/DSEP_transformation.py
iii) Run 

    !python3 UDSEP/code/main_unet.py

Test the trained models:

i) Run 
    !python3 data/test/test_results.py "model" "model_number"

Examples:

    !python3 data/test/test_results.py unet 150

    !python3 data/test/test_results.py cyclegan g_model_AtoB_016894.h5
