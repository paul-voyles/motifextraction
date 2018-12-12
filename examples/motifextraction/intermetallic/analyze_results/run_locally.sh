python create_motifs.py ../data/affinities/

python align_motif.py ../data/averaged/averaged_0.xyz &
python align_motif.py ../data/averaged/averaged_1.xyz &
python align_motif.py ../data/averaged/averaged_2.xyz &
python align_motif.py ../data/averaged/averaged_3.xyz &
python align_motif.py ../data/averaged/averaged_4.xyz &
python align_motif.py ../data/averaged/averaged_5.xyz &
python align_motif.py ../data/averaged/averaged_6.xyz &
wait

python extract_errors.py ../data/averaged/averaged_0.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_1.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_2.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_3.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_4.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_5.xyz 3.6 &
python extract_errors.py ../data/averaged/averaged_6.xyz 3.6 &
wait
