# Datasets

Three public datasets, about 1.5 GB total. None is redistributed here; each
requires accepting its provider's terms. Place them under
`artifact/dataset/` in the layout below, or pass `--dataset-root` to any
`run.sh`.

## CIC-IDS2017 — claims 1, 2, 3, 5, 6

Source: https://www.unb.ca/cic/datasets/ids-2017.html
Download the **MachineLearningCSV** archive (labelled flow CSVs, not the pcaps).

    artifact/dataset/cicids2017/
        Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
        Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
        Friday-WorkingHours-Morning.pcap_ISCX.csv
        Monday-WorkingHours.pcap_ISCX.csv
        Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
        Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
        Tuesday-WorkingHours.pcap_ISCX.csv
        Wednesday-workingHours.pcap_ISCX.csv

Flat, all eight CSVs in one directory. The loader concatenates them and
stratified-downsamples to 200,000 rows.

Verify: the concatenation should give **2,830,743 rows x 79 columns**, and the
downsample **200,001 rows**, 80.3% benign. `run.sh` prints both and stops if
they disagree, since a wrong row count means a different subset and therefore
different numbers.

## UNSW-NB15 — claim 4

Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Only the two partitioned CSVs are needed.

    artifact/dataset/unsw_nb15/
        UNSW_NB15_training-set.csv
        UNSW_NB15_testing-set.csv

Verify: training 82,332 rows (44.9% benign), testing 175,341 rows (31.9%
benign). Note the provider's naming: the file called "training-set" is the
smaller of the two, and we use it as the training split as distributed.

## NSL-KDD — claim 4

Source: https://www.unb.ca/cic/datasets/nsl.html

    artifact/dataset/nsl_kdd/
        KDDTrain+.csv
        KDDTest+.csv

If you download the `.txt` variants, rename them to `.csv`; the loader accepts
either. KDDTest+ has substantial distribution shift relative to KDDTrain+, which
is why absolute accuracy on this dataset is far below the other two for every
method including undefended FedAvg. That is a property of the benchmark.

## Checking

    python artifact/scripts/check_datasets.py

Prints row counts and class balance for whatever is present and flags anything
that does not match the values above.
