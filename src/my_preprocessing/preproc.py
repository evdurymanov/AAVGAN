import os
from collections import Counter
import tensorflow as tf
from common.bio.sequence import Sequence
from Bio.SeqIO.FastaIO import SimpleFastaParser
from common.bio.constants import ID_TO_AMINO_ACID, AMINO_ACID_TO_ID, NON_STANDARD_AMINO_ACIDS
import pandas as pd
import numpy as np

# makeblastdb -in ?.fasta -out db_train -dbtype prot -title "uniprot"
# seqkit seq -M 512 .fasta > ?_512.fasta
# cat ?.fasta | seqkit rmdup -s -o ?_clean.fasta
#  cat ?.fasta | seqkit seq | seqkit stats

def filter_non_standard_amino_acids(data, column = ["sequence"]):
    data = data[~data[column].str.contains("|".join(NON_STANDARD_AMINO_ACIDS))]
    return data

def fasta_to_pandas(path, separator=";"):
    with open(path) as fasta_file:
        identifiers, sequences, titles = [], [], []
        for title, sequence in SimpleFastaParser(fasta_file):
            sequences.append(sequence)
        return pd.DataFrame({"sequence": sequences})

def from_amino_acid_to_id(data, column):
    return data[column].apply(lambda x: [AMINO_ACID_TO_ID[c] for c in x])


def to_int_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=data))


def to_float_feature(data):
    return tf.train.Feature(float_list=tf.train.FloatList(value=data))


def save_as_tfrecords(filename, data, columns=["sequence"], extension="tfrecords"):
    try:
        filename = "{}.{}".format(filename, extension)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index, row in data.iteritems():
                feature = {
                    'label': to_int_feature([int(0)])
                }
                for col_name in columns:
                    value = row
                    if isinstance(value, int):
                        feature[col_name] = to_int_feature([value])
                    elif isinstance(value, float):
                        feature[col_name] = to_float_feature([value])
                    elif not isinstance(value, (list,)) and not isinstance(value, int) and ((value.dtype == np.float32) or (value.dtype == np.float64)):
                        feature[col_name] = to_float_feature(value)
                    else:
                        feature[col_name] = to_int_feature(value)
                        feature['length_' + col_name]:  to_int_feature([len(value)])

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print("Data was stored in {}".format(filename))
    except Exception as e:
        print("Something went wrong went writting in to tfrecords file")
        print("Error is ", str(e))




data = fasta_to_pandas('up_sampling_dataset/1_1_3000.fasta')
data.to_csv("1_1_3000.csv", sep = "\t")
data = data[~data["sequence"].str.contains("|".join(NON_STANDARD_AMINO_ACIDS))]
data = from_amino_acid_to_id(data, "sequence")
#data.to_csv("vp1_new.csv", sep = "\t")
save_as_tfrecords('tfrecords/1_1_3000', data)

"""
data = fasta_to_pandas('uniprot-ec 1.1.1.36 val.fasta')

data = data[~data["sequence"].str.contains("|".join(NON_STANDARD_AMINO_ACIDS))]
data = from_amino_acid_to_id(data, "sequence")
#data.to_csv("vp1_new_val.csv", sep = "\t")
save_as_tfrecords('36val', data)
"""

"""
data_previous = pd.read_csv("train_sequences.csv", sep = "\t")
data_previous = data_previous[~data_previous["sequence"].str.contains("|".join(NON_STANDARD_AMINO_ACIDS))]
data_previous = from_amino_acid_to_id(data_previous, "sequence")
data_previous.to_csv("train_sequences_new.csv", sep = "\t")
save_as_tfrecords('tf_records_test', data_previous)
"""