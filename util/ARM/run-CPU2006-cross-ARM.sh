#!/bin/sh
# Run the CPU2006 benchmarks on ARM. Assumes you have the time utility from
# linux.
# TODO: Combine with the "Gen" python script. Create a portable tool that can
# run benchmarks and generate statistics on arbitrary platforms.

# Number of times the benchmarks should be run.
ITER=3

HOME=/home/ghassan
ARNAME=ziped_benches.tar.xz

# extract benchmarks
echo 'Extracting archive that was copied from grace'
tar xJf $ARNAME

echo "Invoking benchmarks $ITER times"
for ((i=0; i<$ITER; i++));
do
    # perlbench
    #echo "Running perlbench"
    #cp 400.perlbench/exe/perlbench_base..exe 400.perlbench/run/run_base_test_.exe.0000/.
    #cd /home/ghassan/400.perlbench/run/run_base_test_.exe.0000
    #time /bin/sh -c "../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib attrs.pl > attrs.out 2>> attrs.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib gv.pl > gv.out 2>> gv.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib makerand.pl > makerand.out 2>> makerand.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib pack.pl > pack.out 2>> pack.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib redef.pl > redef.out 2>> redef.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib ref.pl > ref.out 2>> ref.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib regmesg.pl > regmesg.out 2>> regmesg.err
    #../run_base_test_.exe.0000/perlbench_base..exe -I. -I./lib test.pl > test.out 2>> test.err"
    #cd $HOME

    # gcc
    #echo "Running gcc"
    #cp 403.gcc/exe/gcc_base..exe 403.gcc/run/run_base_test_.exe.0000/.
    #cd /home/ghassan/403.gcc/run/run_base_test_.exe.0000
    #time /bin/sh -c "../run_base_test_.exe.0000/gcc_base..exe cccp.in -o cccp.s > cccp.out 2>> cccp.err"
    #cd $HOME

    # mcf
    echo "Running mcf"
    cp 429.mcf/exe/mcf_base..exe 429.mcf/run/run_base_test_.exe.0000/.
    cd /home/ghassan/429.mcf/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/mcf_base..exe inp.in > inp.out 2>> inp.err"
    cd $HOME

    # sphinx3
    #echo "Running sphinx3"
    #cp 482.sphinx3/exe/sphinx_livepretend_base..exe 482.sphinx3/run/run_base_test_.exe.0000/.
    #cd /home/ghassan/482.sphinx3/run/run_base_test_.exe.0000
    #time /bin/sh -c "../run_base_test_.exe.0000/sphinx_livepretend_base..exe ctlfile . args.an4 > an4.log 2>> an4.err"
    #cd $HOME

    # milc
    echo "Running milc"
    cp 433.milc/exe/milc_base..exe 433.milc/run/run_base_test_.exe.0000/.
    cd /home/ghassan/433.milc/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/milc_base..exe < su3imp.in > su3imp.out 2>> su3imp.err"
    cd $HOME

    # sjeng
    echo "Running sjeng"
    cp 458.sjeng/exe/sjeng_base..exe 458.sjeng/run/run_base_test_.exe.0000/.
    cd /home/ghassan/458.sjeng/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/sjeng_base..exe test.txt > test.out 2>> test.err"
    cd $HOME

    # libquantum
    echo "Running libquantum"
    cp 462.libquantum/exe/libquantum_base..exe 462.libquantum/run/run_base_test_.exe.0000/.
    cd /home/ghassan/462.libquantum/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/libquantum_base..exe 337 6 > test.out 2>> test.err"
    cd $HOME

    # soplex
    #echo "Running soplex"
    #cp 450.soplex/exe/soplex_base..exe 450.soplex/run/run_base_test_.exe.0000/.
    #cd /home/ghassan/450.soplex/run/run_base_test_.exe.0000
    #time /bin/sh -c "../run_base_test_.exe.0000/soplex_base..exe -m100000 test.mps > test.out 2>> test.stderr"
    #cd $HOME

    # povray
    echo "Running povray"
    cp 453.povray/exe/povray_base..exe 453.povray/run/run_base_test_.exe.0000/.
    cd /home/ghassan/453.povray/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/povray_base..exe SPEC-benchmark-test.ini > SPEC-benchmark-test.stdout 2>> SPEC-benchmark-test.stderr"
    cd $HOME

    # omnetpp
    echo "Running omnetpp"
    cp 471.omnetpp/exe/omnetpp_base..exe 471.omnetpp/run/run_base_test_.exe.0000/.
    cd /home/ghassan/471.omnetpp/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/omnetpp_base..exe omnetpp.ini > omnetpp.log 2>> omnetpp.err"
    cd $HOME

    # astar
    echo "Running astar"
    cp 473.astar/exe/astar_base..exe 473.astar/run/run_base_test_.exe.0000/.
    cd /home/ghassan/473.astar/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/astar_base..exe lake.cfg > lake.out 2>> lake.err"
    cd $HOME

    # xalancbmk
    #echo "Running xalancbmk"
    #cp 483.xalancbmk/exe/Xalan_base..exe 483.xalancbmk/run/run_base_test_.exe.0000/.
    #cd /home/ghassan/483.xalancbmk/run/run_base_test_.exe.0000
    #time /bin/sh -c "../run_base_test_.exe.0000/Xalan_base..exe -v test.xml xalanc.xsl > test.out 2>> test.err"
    #cd $HOME

    # dealII
    echo "Running dealII"
    cp 447.dealII/exe/dealII_base..exe 447.dealII/run/run_base_test_.exe.0000/.
    cd /home/ghassan/447.dealII/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/dealII_base..exe 8 > log 2>> dealII.err"
    cd $HOME

    # bzip2
    echo "Running bzip2"
    cp 401.bzip2/exe/bzip2_base..exe 401.bzip2/run/run_base_test_.exe.0000/.
    cd /home/ghassan/401.bzip2/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/bzip2_base..exe input.program 5 > input.program.out 2>> input.program.err
    ../run_base_test_.exe.0000/bzip2_base..exe dryer.jpg 2 > dryer.jpg.out 2>> dryer.jpg.err"
    cd $HOME

    # namd
    echo "Running namd"
    cp 444.namd/exe/namd_base..exe 444.namd/run/run_base_test_.exe.0000/.
    cd /home/ghassan/444.namd/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/namd_base..exe --input namd.input --iterations 1 --output namd.out  > namd.stdout 2>> namd.err"
    cd $HOME

    # gobmk
    echo "Running gobmk"
    cp 445.gobmk/exe/gobmk_base..exe 445.gobmk/run/run_base_test_.exe.0000/.
    cd /home/ghassan/445.gobmk/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < capture.tst > capture.out 2>> capture.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < connect.tst > connect.out 2>> connect.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < connect_rot.tst > connect_rot.out 2>> connect_rot.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < connection.tst > connection.out 2>> connection.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < connection_rot.tst > connection_rot.out 2>> connection_rot.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < cutstone.tst > cutstone.out 2>> cutstone.err
    ../run_base_test_.exe.0000/gobmk_base..exe --quiet --mode gtp < dniwog.tst > dniwog.out 2>> dniwog.err"
    cd $HOME

    # hmmer
    echo "Running hmmer"
    cp 456.hmmer/exe/hmmer_base..exe 456.hmmer/run/run_base_test_.exe.0000/.
    cd /home/ghassan/456.hmmer/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/hmmer_base..exe --fixed 0 --mean 325 --num 45000 --sd 200 --seed 0 bombesin.hmm > bombesin.out 2>> bombesin.err"
    cd $HOME

    # h264ref
    echo "Running h264ref"
    cp 464.h264ref/exe/h264ref_base..exe 464.h264ref/run/run_base_test_.exe.0000/.
    cd /home/ghassan/464.h264ref/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/h264ref_base..exe -d foreman_test_encoder_baseline.cfg > foreman_test_baseline_encodelog.out 2>> foreman_test_baseline_encodelog.err"
    cd $HOME

    # lbm
    echo "Running lbm"
    cp 470.lbm/exe/lbm_base..exe 470.lbm/run/run_base_test_.exe.0000/.
    cd /home/ghassan/470.lbm/run/run_base_test_.exe.0000
    time /bin/sh -c "../run_base_test_.exe.0000/lbm_base..exe 20 reference.dat 0 1 100_100_130_cf_a.of > lbm.out 2>> lbm.err"
    cd $HOME
done # run benchmarks
