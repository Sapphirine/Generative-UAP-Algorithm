# Generative Universal Adversarial Perturbation (UAP) Algorithm

The key feature of this source code can be found here: Generative-UAP-Algorithm/GenerateUAP/MakeGenerativeUaps.py
This python script implements a generative UAP algorithm that is independent of both data and models.

The other python scripts in Generative-UAP-Algorithm/GenerateUAP/ were used for analysis and development of the generative UAP algorithm. Note that, to run these other scripts, you need access to the files in Generative-UAP-Algorithm/data/ and also must download the ILSVRC2012 validation set (https://academictorrents.com/collection/imagenet-2012).

In Generative-UAP-Algorithm/data/, you will also find the 44 Model+Date-Dependent UAPs and the 50 Generative UAPs used to get the results in the paper. Additionally, you will find BQ-Data.zip which contains the 4.7M rows of data that must be hosted on Google Cloud Big Query for the django functionality.

In Generative-UAP-Algorithm/django_website/UAP/, you will find the django and HTML source code used to generate some of the figures in the paper as well as a demo website. To get this working, you need access to the ILSVRC2012 validation set stored in a Google Cloud bucket as well as the Big Query data (in Generative-UAP-Algorithm/data/). 
