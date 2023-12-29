# SMsplice

### Dependencies
Begin by cloning and entering repository

	git clone https://github.com/kmccue/SMsplice/
 	cd SMsplice

Install environment **(currently linux-specific)** with:

    conda env create -f environment.yml

Follow instructions on https://github.com/mennthor/awkde and use -e flag when installing:

	conda activate SMsplice
	pip install -e ./awkde

### Example Calls
Note: requires genome fastas to be downloaded, and paths indicated by ** to be changed

*To run SMsplice on Arabidopsis sequences with pre-learned SRE scores ( requires awkde from https://github.com/mennthor/awkde ) and print individual predictions:*

	python runSMsplice.py -c ./canonical_datasets/canonical_dataset_TAIR10.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g **/path/to/TAIR10.fa** -m ./maxEnt_models/arabidopsis/ --prelearned_sres arabidopsis --print_predictions

*To learn real vesus decoy seeded CASS on Arabidopsis seqeunces:*

	python runCASS.py -c ./canonical_datasets/canonical_dataset_TAIR10.txt -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g **/path/to/TAIR10.fa** -m ./maxEnt_models/arabidopsis/ --learning_seed real-decoy --learn_sres 

*To train new Arabidopsis MaxEnt models:*

	python trainMaxEnt.py -a ./allSS_datasets/allSS_dataset_TAIR10.txt -g **/path/to/TAIR10.fa** -m **/output/directory/**

### Modifications
To change test set, edit line 85 of runSMsplice.py or runCASS.py\
To change structural parameters, edit lines 135-170 of runSMsplice.py

