#!/bin/sh
sbatch -N 1 ./run-spark/spark-launcher-manuLux.sh
sbatch -N 1 ./run-spark/spark-launcher-manuBBC.sh
sbatch -N 1 ./run-spark/spark-launcher-manuCNN.sh
sbatch -N 1 ./run-spark/spark-launcher-manuNYT.sh
sbatch -N 1 ./run-spark/spark-launcher-autoLux.sh
sbatch -N 1 ./run-spark/spark-launcher-autoBBC.sh
sbatch -N 1 ./run-spark/spark-launcher-autoCNN.sh
sbatch -N 1 ./run-spark/spark-launcher-autoNYT.sh