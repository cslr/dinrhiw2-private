#!/bin/sh

rm -f wdbc-test.ds

# creates training dataset for nntool

./dstool -create wdbc-test.ds
./dstool -create:30:input wdbc-test.ds
./dstool -create:1:output wdbc-test.ds
./dstool -list wdbc-test.ds
./dstool -import:0 wdbc-test.ds wdbc.in
./dstool -import:1 wdbc-test.ds wdbc.out
./dstool -padd:0:meanvar wdbc-test.ds
## ./dstool -padd:0:pca wdbc-test.ds
./dstool -padd:1:meanvar wdbc-test.ds

./dstool -list wdbc-test.ds

# ARCH="30-30-30-30-30-1"
ARCH="30-60-60-60-60-1"
# uses nntool trying to learn from dataset

rm -f wdbcnn.cfg

# ./nntool -v --negfb --time 400 wdbc-test.ds $ARCH wdbcnn.cfg random
# ./nntool -v wdbc-test.ds $ARCH wdbcnn.cfg lbfgs
./nntool -v --time 600 wdbc-test.ds $ARCH wdbcnn.cfg grad

##################################################
# testing

./nntool -v wdbc-test.ds $ARCH wdbcnn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f wdbc-test.ds wdbc-pred.ds
./dstool -clear:1 wdbc-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v wdbc-pred.ds $ARCH wdbcnn.cfg use

./dstool -list wdbc-test.ds
./dstool -list wdbc-pred.ds

./dstool -print:1 wdbc-pred.ds
tail wdbc.out


