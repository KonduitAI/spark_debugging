package org.deeplearning4j;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by susaneraly on 2/11/20.
 */
@Slf4j
public class SparkTestImport {

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String masterIP;

    @Parameter(names = {"--netmask"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String netmask = "255.255.255.0";


    public static void main(String[] args) throws Exception {
        try {
            new SparkTestImport().run(args);
        } catch (Throwable t) {
            t.printStackTrace();
            throw t;
        }
    }

    public void run(String[] args) throws Exception {
        JCommander.newBuilder().addObject(this).build().parse(args);

        int minibatch = 8;
        final int NINSIZE = 500;

        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                //.unicastPort(12345)
                .controllerAddress(masterIP)
                .networkMask(netmask)
                .build();

        SharedTrainingMaster trainingMaster = new SharedTrainingMaster.Builder(voidConfiguration, 1)
                .thresholdAlgorithm(new AdaptiveThresholdAlgorithm())
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .workersPerNode(1)
                .batchSizePerWorker(minibatch)
                .build();

        //final String MODEL_PATH = new ClassPathResource("proof_of_concept_model.h5").getFile().getAbsolutePath();
        //final String MODEL_PATH = new ClassPathResource("proof_of_concept_model_highlr.h5").getFile().getAbsolutePath();
        final String MODEL_PATH = new ClassPathResource("proof_of_concept_model_lowlr.h5").getFile().getAbsolutePath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(MODEL_PATH, true);
        System.out.println(model.summary());

        JavaSparkContext sc = new JavaSparkContext();
        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, model, trainingMaster);

        int nMinibatches = 100;
        int seqLengthMin = 32;
        int seqLengthMax = 100;

        List<DataSet> data = new ArrayList<>();
        Random r = new Random(12345);
        for (int i = 0; i < nMinibatches; i++) {
            int length = seqLengthMin + r.nextInt(seqLengthMax - seqLengthMin);
            INDArray f = Nd4j.rand(DataType.FLOAT, minibatch, NINSIZE, length);
            //Output is the average of all features at the last time step
            INDArray l = f.sum(1).getColumns((int)f.size(2)-1).div(f.size(1));
            data.add(new DataSet(f, l));
        }

        JavaRDD<DataSet> rdd = sc.parallelize(data);

        log.info("About to start training...");
        long start = System.currentTimeMillis();
        int epochs = Integer.MAX_VALUE;
        for (int i = 0; i < epochs; i++) {
            log.info("Starting epoch {}, start + {}", i, time(start));
            net.fit(rdd);
        }

        log.info("Done");
    }

    private String time(long start) {
        return DurationFormatUtils.formatDurationHMS(System.currentTimeMillis() - start);
    }

}

