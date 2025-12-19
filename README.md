# LARGAL
## By Aditya Anand

LARGAL is an AI-detection and classification for detecting and classifying radio galaxies. It uses YOLO v8s to find radio galaxies, a variational autoencoder to encode radio galaxies in a smaller latent space, and a histogram gradient boosting model is trauned on the latent representations of the VAE.

The finetuned YOLO model is found in yolov8s-finetune. In order to finetune, change the dataset.yaml path to your project's pat and run the finetune_yolo.ipynb notebook.

To finetune the VAE, you need to first crop out the radio galaxies from the dataset by running docrops.py, which will create three pickle files with cropped data in them. Youu can then run the clusterizeVAE.ipynb notebook to create a pytorch VAE saved in VAEModelCubicFit.pth.

After the VAE is created, you can extract latent vectors from your crops with the script in extract_latents.py. This will create files in the directory latents/ representing the latent vectors, areas, and their classifications.

The classifiers can be trained with the train_classifier_multisize, which trains two sets of classifiers: one is small (area < 24^2 pixels) and the other is medium (area > 24^2 pixels). It will test several samples and save them in classifiers/.

The whole pipeline can be evaluated with evaluate_bboxes_multisize.py to produce an mAP score. The compute_map is a helper library. The t-SNE plot was created with run_tsne_latents_valcubic.py program.