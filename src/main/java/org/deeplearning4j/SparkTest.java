package org.deeplearning4j;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Slf4j
public class SparkTest {

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String masterIP;

    @Parameter(names = {"--netmask"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String netmask = "255.255.255.0";


    public static void main(String[] args) throws Exception {
        try {
            new SparkTest().run(args);
        } catch (Throwable t){
            t.printStackTrace();
            throw t;
        }
    }

    public void run(String[] args) throws Exception {
        JCommander.newBuilder().addObject(this).build().parse(args);

        int minibatch = 8;
        int nInOutSize = 64;

        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .unicastPort(12345) // These DLConfig values are determined at runtime
                .controllerAddress(masterIP)
                .networkMask(netmask)
                .build();

        SharedTrainingMaster trainingMaster = new SharedTrainingMaster.Builder(voidConfiguration, 1)
                .thresholdAlgorithm(new AdaptiveThresholdAlgorithm())
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .workersPerNode(1)
                .batchSizePerWorker(minibatch)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new LSTM.Builder().nIn(nInOutSize).nOut(nInOutSize).build())
                .layer(new LSTM.Builder().nIn(nInOutSize).nOut(nInOutSize).build())
                .layer(new RnnOutputLayer.Builder().nIn(nInOutSize).nOut(nInOutSize).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .tBPTTLength(32)
                .build();

        JavaSparkContext sc = new JavaSparkContext();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, conf, trainingMaster);

        int nMinibatches = 200;
        int seqLengthMin = 32;
        int seqLengthMax = 256;

        List<DataSet> data = new ArrayList<>();
        Random r = new Random(12345);
        for( int i=0; i<nMinibatches; i++ ){
            int length = seqLengthMin + r.nextInt(seqLengthMax - seqLengthMin);
            INDArray f = Nd4j.rand(DataType.FLOAT, minibatch, nInOutSize, length);
            INDArray l = Nd4j.zeros(DataType.FLOAT, minibatch, nInOutSize, length);
            for( int e=0; e<minibatch; e++ ){
                for( int t=0; t<length; t++ ){
                    int idx = r.nextInt(nInOutSize);
                    l.putScalar(e, idx, t, 1.0);
                }
            }
            data.add(new DataSet(f, l));
        }

        JavaRDD<DataSet> rdd = sc.parallelize(data);

        log.info("About to start training...");
        long start = System.currentTimeMillis();
        int epochs = Integer.MAX_VALUE;
        for( int i=0; i<epochs; i++ ){
            log.info("Starting epoch {}, start + {}", i, time(start));
            net.fit(rdd);
        }

        log.info("Done");
    }

    private String time(long start){
        return DurationFormatUtils.formatDurationHMS(now - start);
    }

}
