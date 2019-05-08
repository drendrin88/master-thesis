#!/bin/sh
cd /Volumes/Card/master-thesis/manual/lux
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/manual/bbc
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/manual/cnn
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/manual/nyt
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/auto/lux
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/auto/bbc
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/auto/cnn
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q
cd /Volumes/Card/master-thesis/auto/nyt
spark-shell -i /Volumes/Card/master-thesis/spider-v1.6.scala
:q