from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio import pairwise2
from Bio import Align

capsid = []
capsid_names = []
capsid_lengths = []
capsidun = []
parvocapun = []
vp1un = []
vp2un = []
vp3un = []


with open("pashas5121024.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        capsid.append(seq)
        capsid_names.append(title)
        capsid_lengths.append(len(seq))
"""
with open("capsid_unique.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        capsidun.append(seq)
with open("Parvo-cap-unique.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        parvocapun.append(seq)
with open("vp1_unique.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        vp1un.append(seq)
with open("vp2_unique.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        vp2un.append(seq)
with open("vp3_unique.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        vp3un.append(seq)
"""
print(len(capsid), sum(capsid_lengths) / len(capsid_lengths))