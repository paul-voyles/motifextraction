cd ../data
tar -czf clusters.tar.gz clusters/
cd ../alignment

python create_local_run_file.py
chmod u+x run_all.sh
./run_all.sh
