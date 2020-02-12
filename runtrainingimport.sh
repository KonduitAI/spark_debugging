#!/usr/bin/env bash

#REQUIRED arguments: Set these before running
MASTER_IP=10.0.2.4                                 #IP address of master node
MASTER_PORT=7077 
NETWORK_MASK=10.0.2.0/16                                #Network maske For example, 10.0.0.0/16


#Optional argumenst: Set these only if the defaults aren't suitable
SPARKSUBMIT=/opt/spark/bin/spark-submit

# For memory config, see https://deeplearning4j.org/memory
JAVA_HEAP_MEM=8G
OFFHEAP_MEM_JAVACPP=12G
OFFHEAP_JAVACPP_MAX_PHYS=20G
#Aeron buffer. Default of 32MB is fine for this example. Larger neural nets may require larger: 67108864 or 134217728. Must be a power of 2 exactly
AERON_BUFFER=33554432

#Other variables. Don't modify these
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/target/deeplearning4j-examples-1.0.0-beta5-bin.jar


CMD="${SPARKSUBMIT}
    --class org.deeplearning4j.SparkTestImport
    --conf spark.locality.wait=0
    --conf 'spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --conf 'spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --driver-java-options '-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --master spark://${MASTER_IP}:${MASTER_PORT}
    --deploy-mode client
    ${JARFILE}
    --masterIP ${MASTER_IP}
    --netmask ${NETWORK_MASK}
    "

eval $CMD
