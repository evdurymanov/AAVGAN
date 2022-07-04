###get matrix distance
./clustalo -i .fasta --distmat-out=.mat --force --full

###post matrix distance refactoring
sed 's/^ *//' .mat | cut -d" " -f2- > .mat   

###make database using blast
makeblastdb -in ?.fasta -out db_train -dbtype prot -title "uniprot"

###filtering by length seqkit
seqkit seq -M 512 .fasta > ?_512.fasta

###remove duplicates using seqkit
cat ?.fasta | seqkit rmdup -s -o ?_clean.fasta

###seqkit stats
cat ?.fasta | seqkit seq | seqkit stats

