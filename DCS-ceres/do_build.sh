cd save
rm -rf *

cd ..
mkdir build
cd build
rm -rf *
cmake ..
make -j8
# Forward all CLI args to the binary so extra flags like --online work
./main "$@"

cd ..

cd drawer
./do_plot.sh
cd ..
