# rm -rf build
# mkdir build
# cd build
# cmake ..
# make -j 24
# cd ..

# cd build
# make -j 24
# cd ..
# ./run_nts.sh 1 configs/vary_edges/80M.cfg > ./log/random/80M.log
./run_nts.sh 1 configs/vary_edges/160M.cfg > ./log/random/160M.log
./run_nts.sh 1 configs/vary_edges/320M.cfg > ./log/random/320M.log

./run_nts.sh 1 configs/vary_dim/256.cfg > ./log/random/256.log
./run_nts.sh 1 configs/vary_dim/512.cfg > ./log/random/512.log
./run_nts.sh 1 configs/vary_dim/1024.cfg > ./log/random/1024.log

./run_nts.sh 1 configs/vary_label/16.cfg > ./log/random/16.log
./run_nts.sh 1 configs/vary_label/32.cfg > ./log/random/32.log
./run_nts.sh 1 configs/vary_label/64.cfg > ./log/random/64.log

