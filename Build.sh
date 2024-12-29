rm -rf build
mkdir build
cd build
cmake ..
make -j 24
cd ..

cd build
make -j 24
cd ..

./run_nts.sh 1 APPNP.cfg > ./log/debug.log
# ./run_nts.sh 1 APPNP.cfg
echo "finish"
